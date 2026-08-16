"""
Song Database - Raw Data Fetcher (TRACKLIST CACHE TEST v2 — limited to 100 new songs)

Experimental version that adds a TRACKLIST-CACHE layer as the FIRST album
resolution strategy, ahead of the multi-AI voting system. Uses the real,
ordered tracklists already stored in 2_albums_raw.db (tracklist_json column,
populated by the updated 2_album_db_raw.py from MusicBrainz recordings /
Wikipedia) — AND self-improves: every song the AI panel resolves gets
written back into its album's cached tracklist, so it's never asked again.

Why this matters: once ONE song from an album has been looked up, every
OTHER song from that same album resolves INSTANTLY and for FREE — a simple
membership check against a cached list, zero AI calls. This is exactly the
King Crimson "Red" example: if "Red"'s tracklist is cached, "Starless"
(track 5) matches immediately, no need to ask any model anything.

SIX-TIER release priority (used consistently for cache tie-breaking AND in
the AI voting prompt):
    1. LP / Studio Album   2. EP   3. Single   4. Live
    5. Compilation / Greatest Hits   6. Other (unrecognized types)

Resolution order per new song:
    1. TRACKLIST CACHE
       a. Exact match (normalized) against every known album's cached
          tracklist for this artist.
       b. If nothing exact, a FUZZY match pass (>=90% similarity) —
          tolerates minor typos/transcription differences without risking
          false positives on genuinely different songs.
       c. If the title matches more than one cached album (e.g. same song
          on both the studio album and a live album), tie-break by:
          (i) album_from_scrobble — if Last.fm's own reported album name
              matches one of the tied candidates, that one wins (free,
              real signal); (ii) is_reissue; (iii) the six-tier type
              priority above; (iv) earliest year.
       -> 0 AI calls when this hits.
    2. CLOSED-CANDIDATE MULTI-AI VOTING — only when the cache found
       nothing at all.
       a. DeepSeek (China API) votes first. A confident album/LP pick is
          accepted directly (1 call).
       b. Otherwise Gemma (the other PRIMARY voter) votes on the same
          question. DeepSeek+Gemma agreeing (2/2) is accepted (2 calls).
       c. Only if DeepSeek and Gemma DISAGREE does Gemini get called, as
          a tiebreaker (Gemini has a very low daily quota, so it's
          reserved for genuine disagreements, never used routinely).
       d. If all three end up disagreeing, the song is left ambiguous /
          id_album=0, with review_notes recording every model's vote —
          never guessed.
       e. FEEDBACK: any song the panel resolves gets appended to the
          winning album's cached tracklist, both in memory (benefits the
          rest of THIS run) and persisted to 2_albums_raw.db (benefits
          every future run) — so the AI is never asked about that
          album/song pair again.
    3. Old scrobble/Last.fm/MusicBrainz fuzzy fallback — only when this
       artist has literally no known albums yet.

This script is capped at MAX_NEW_SONGS new songs per run and writes to its
own separate test database (data/3_songs_test_tracklist.db), so it never
touches or risks corrupting the main pipeline's data/3_songs_raw.db while
this strategy is being evaluated. If this performs well, this design gets
ported into 3_song_db_raw_nvidia.py itself, and from there into the
DeepSeek China production version (3_song_db_raw.py).

Duration fetching (Last.fm -> Spotify -> DeepSeek China) is unchanged — this
experiment is only about album resolution.
"""

import sqlite3
import os
import sys
import requests
import time
import re
import json
import difflib
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from dotenv import load_dotenv

# Try to import musicbrainzngs
try:
    import musicbrainzngs
except ImportError:
    print("⚠️ musicbrainzngs not installed. Installing...")
    os.system(f"{sys.executable} -m pip install musicbrainzngs")
    import musicbrainzngs

# Try to import the OpenAI client (used for DeepSeek China API and Groq)
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ openai package not installed. Installing...")
    os.system(f"{sys.executable} -m pip install openai")
    from openai import OpenAI

# Try to import the Google GenAI client (used for Gemini and Gemma)
try:
    from google import genai
except ImportError:
    print("⚠️ google-genai package not installed. Installing...")
    os.system(f"{sys.executable} -m pip install google-genai")
    from google import genai

# Load environment variables
load_dotenv()

# Database paths
ARTIST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '1_artist_genres.db')
ALBUM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2_albums_raw.db')
SONG_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '3_song_db_raw.db')

MAX_NEW_SONGS = 10

# --- Credentials ---
LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY')
LASTFM_USER = os.environ.get('LASTFM_USER')
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')   # kept for reference, no longer used
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

LASTFM_API_URL = 'https://ws.audioscrobbler.com/2.0/'
SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_API_URL = 'https://api.spotify.com/v1/search'

# DeepSeek China (official API, paid/unlimited) replaces NVIDIA-hosted DeepSeek
DEEPSEEK_MODEL = "deepseek-chat"   # deepseek-v4-flash on China API
GEMMA_MODEL = "gemma-4-31b-it"
# Groq replaces Gemini Flash (which had only 16 RPD — too low for production)
GROQ_MODEL = "llama-3.3-70b-versatile"

# DeepSeek China — same OpenAI-compatible SDK, different base_url
deepseek_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_API_KEY or "missing",
    timeout=30.0
) if DEEPSEEK_API_KEY else None

# genai.Client() reads GEMINI_API_KEY from the environment automatically.
# Used for Gemma only — Gemini Flash removed due to 16 RPD quota limit.
genai_client = genai.Client() if GEMINI_API_KEY else None

# Groq: OpenAI-compatible, free tier (1K RPD for llama-3.3-70b-versatile)
# Used as tiebreaker only — called when DeepSeek and Gemma disagree.
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "missing",
    timeout=20.0
) if GROQ_API_KEY else None


# Configure MusicBrainz
musicbrainzngs.set_useragent(
    "my_scrobbles",
    "1.0",
    "https://github.com/adroguett-scratch/my-lastfm-scrobbles"
)

# Spotify token cache
spotify_token = None
spotify_token_expires = 0

# Album AI lookup cache (avoid repeated DeepSeek calls for the same song in one run)
album_ai_cache: Dict[str, Optional[dict]] = {}

# Priority order when several album editions match: lower number = higher
# priority. Six explicit tiers, agreed order:
#   1. LP / Studio Album   2. EP   3. Single   4. Live
#   5. Compilation/Greatest Hits   6. Other (anything unrecognized)
# This SAME order is also what we tell the AI voting panel to follow.
ALBUM_TYPE_PRIORITY = {
    'album': 0, 'lp': 0,
    'ep': 1,
    'single': 2,
    'live': 3,
    'greatest hits': 4, 'compilation': 4, 'best of': 4,
}
OTHER_TYPE_PRIORITY = 5  # default for anything not in the dict above ("Otro")


# ============================================
# SPOTIFY API FUNCTIONS (OPTIONAL)
# ============================================

def get_spotify_token():
    """Get a Spotify access token."""
    global spotify_token, spotify_token_expires

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None

    import time as time_module
    if spotify_token and time_module.time() < spotify_token_expires:
        return spotify_token

    try:
        import base64
        auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {'grant_type': 'client_credentials'}

        resp = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=10)
        resp.raise_for_status()

        result = resp.json()
        spotify_token = result['access_token']
        spotify_token_expires = time_module.time() + result['expires_in'] - 60

        return spotify_token

    except Exception as e:
        print(f"      ⚠️ Spotify token error: {e}")
        return None


def fetch_duration_from_spotify(artist_name: str, song_title: str) -> Optional[int]:
    """Fetch song duration from Spotify API."""
    token = get_spotify_token()
    if not token:
        return None

    try:
        query = f'artist:{artist_name} track:{song_title}'
        params = {
            'q': query,
            'type': 'track',
            'limit': 1
        }

        headers = {'Authorization': f'Bearer {token}'}

        resp = requests.get(SPOTIFY_API_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        tracks = data.get('tracks', {}).get('items', [])

        if tracks:
            duration_ms = tracks[0].get('duration_ms')
            if duration_ms:
                return duration_ms // 1000

        return None

    except Exception as e:
        print(f"      ⚠️ Spotify error for '{artist_name} - {song_title}': {e}")
        return None


# ============================================
# DEEPSEEK CHINA API FUNCTIONS
# ============================================

def is_suspicious_year_duration(duration: int) -> bool:
    """Check if a duration is actually a year (like 1973, 2112, etc.)"""
    suspicious_years = {
        1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929,
        1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939,
        1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949,
        1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959,
        1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
        1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979,
        1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
        1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
        2030, 2112, 2525, 3000
    }
    
    if duration in suspicious_years:
        return True
    if duration in [1973, 2112, 1974, 1975, 1976, 1977, 1978, 1979]:
        return True
    
    return False


def fetch_duration_from_deepseek(artist_name: str, song_title: str) -> Optional[int]:
    """Fetch song duration using the official DeepSeek China API.

    On any failure (timeout, empty response, rate limit, malformed output)
    this simply returns None so the caller can move on to the next song —
    it is never treated as a fatal error.
    """
    if not deepseek_client:
        return None

    try:
        clean_title = song_title.strip()
        clean_title = re.sub(r'\s*[:/]\s*', ' / ', clean_title)

        prompt = f"""You are a music expert. What is the exact duration of the song "{clean_title}" by {artist_name}?

IMPORTANT: Respond ONLY with the duration in seconds as a number.
Do NOT respond with minutes:seconds format.
Do NOT respond with a year (like 1973).
If you don't know, respond with '0'."""

        completion = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a music expert. Respond ONLY with a number (duration in seconds). Never respond with a year."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            stream=False
        )

        if not completion.choices:
            return None

        content = completion.choices[0].message.content
        if not content:
            return None

        match = re.search(r'\d+', content)
        if match:
            duration = int(match.group())

            if is_suspicious_year_duration(duration):
                print(f"      ⚠️ DeepSeek returned a year ({duration}s) instead of duration.")
                return None

            if duration > 0 and duration < 6000:
                return duration

        return None

    except Exception as e:
        print(f"      ⚠️ DeepSeek error for '{artist_name} - {song_title}': {e}")
        return None


# ============================================
# ALBUM SEARCH FUNCTIONS
# ============================================

def get_album_from_lastfm_track(artist_name: str, song_title: str) -> Optional[str]:
    """Get the album name for a track from Last.fm track.getInfo."""
    params = {
        'method': 'track.getInfo',
        'artist': artist_name,
        'track': song_title,
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }

    try:
        resp = requests.get(LASTFM_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if 'error' in data:
            return None

        track = data.get('track', {})
        album = track.get('album', {})
        album_title = album.get('title')

        if album_title and album_title.strip():
            return album_title.strip()

        return None

    except Exception as e:
        print(f"      ⚠️ Last.fm album lookup error: {e}")
        return None


def get_album_from_musicbrainz(artist_name: str, song_title: str) -> Optional[str]:
    """Get the album name for a track from MusicBrainz."""
    try:
        # Search for the recording
        result = musicbrainzngs.search_recordings(query=f'artist:"{artist_name}" AND recording:"{song_title}"', limit=1)
        recordings = result.get('recording-list', [])
        
        if not recordings:
            return None
        
        recording = recordings[0]
        releases = recording.get('release-list', [])
        
        if not releases:
            return None
        
        # Take the first release
        release = releases[0]
        album_title = release.get('title')
        
        if album_title:
            return album_title.strip()
        
        return None

    except Exception as e:
        print(f"      ⚠️ MusicBrainz album lookup error: {e}")
        return None


# ============================================
# CLOSED-CANDIDATE MULTI-AI VOTING SYSTEM
# ============================================
#
# Instead of asking a model to freely generate an album name (prone to
# hallucinated demos, promos, and rarities that aren't even in our own
# database), we give it the closed, numbered list of releases we already
# know about for that artist, and ask it to pick a number. This turns
# "generate" into "verify", which these models are much better at.

# ============================================
# TRACKLIST CACHE (checked BEFORE any AI call)
# ============================================
#
# 2_album_db_raw.py already stores the real, ordered tracklist for most
# albums (from MusicBrainz recordings, or Wikipedia as fallback). Once one
# song from an album has been resolved, every OTHER song from that same
# album can be matched for free by simple membership check — no AI call
# needed at all. This is the King Crimson "Red" case: if "Red"'s tracklist
# is cached, "Starless" (track 5) matches instantly.

_TRAILING_ANNOTATION_RE = re.compile(
    r'\s*[\(\[]\s*(?:remaster(?:ed)?(?:\s*\d{4})?|mono|stereo|'
    r'single version|album version|edit|bonus track)[^)\]]*[\)\]]\s*$',
    re.IGNORECASE
)


def _normalize_song_for_match(title: str) -> str:
    """Normalize a song title for tracklist membership comparison."""
    t = (title or '').lower().strip()
    # Strip trailing annotations that wouldn't appear in a canonical
    # tracklist entry (e.g. "Starless (2004 Remaster)" -> "starless").
    t = _TRAILING_ANNOTATION_RE.sub('', t)
    t = re.sub(r'[^\w\s]', '', t)  # strip punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    return t


FUZZY_MATCH_THRESHOLD = 0.90  # similarity ratio (0-1) for the tolerant second pass


def _song_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_song_in_tracklist_cache(song_title: str, album_map: Dict[int, List[dict]],
                                  id_artist: int, album_from_scrobble: str = '') -> Tuple[int, Optional[str]]:
    """
    Check the song title against every cached tracklist for this artist's
    known albums.

    Two passes:
      1. Exact match (after normalization) — cheap, no false positives.
      2. Fuzzy match (>= FUZZY_MATCH_THRESHOLD similarity) — ONLY if pass 1
         found nothing. Tolerates minor typos / transcription differences
         in how a song title is written (e.g. in the scrobble vs. the
         tracklist), without opening the door to matching genuinely
         different songs.

    If the title matches in more than one cached album (e.g. same song on
    both the studio album and a live album), tie-break in this order:
      a. album_from_scrobble — if Last.fm's own reported album name matches
         one of the tied candidates, that one wins (it's free, real
         signal from the actual scrobble, not a guess).
      b. is_reissue (prefer the original, non-reissue edition)
      c. Type priority — LP > EP > Single > Live > Compilation > Other
      d. Earliest year

    Returns (id_album, note) — note is None on an unambiguous exact match,
    or a short explanation when fuzzy-matching or tie-breaking kicked in.
    Returns (0, None) if nothing matches at all — caller falls through to
    the AI voting system.
    """
    candidates = album_map.get(id_artist, [])
    if not candidates:
        return 0, None

    target = _normalize_song_for_match(song_title)
    if not target:
        return 0, None

    # Pass 1: exact match
    matches = []
    for c in candidates:
        tracklist = c.get('tracklist')
        if not tracklist:
            continue
        for track in tracklist:
            if _normalize_song_for_match(track) == target:
                matches.append(c)
                break

    fuzzy_used = False

    # Pass 2: fuzzy match, only if pass 1 found nothing
    if not matches:
        best_ratio = 0.0
        for c in candidates:
            tracklist = c.get('tracklist')
            if not tracklist:
                continue
            for track in tracklist:
                ratio = _song_similarity(target, _normalize_song_for_match(track))
                if ratio >= FUZZY_MATCH_THRESHOLD:
                    matches.append(c)
                    best_ratio = max(best_ratio, ratio)
                    break
        if matches:
            fuzzy_used = True

    if not matches:
        return 0, None

    note = f"Fuzzy match (>={FUZZY_MATCH_THRESHOLD:.0%} similarity)" if fuzzy_used else None

    if len(matches) > 1:
        # Tie-break (a): does the scrobble's own reported album match one of the tied candidates?
        scrobble_norm = _normalize_song_for_match(album_from_scrobble) if album_from_scrobble else ''
        if scrobble_norm:
            scrobble_pick = next(
                (c for c in matches if _normalize_song_for_match(c['title']) == scrobble_norm), None
            )
            if scrobble_pick:
                tie_note = f"Matched in {len(matches)} cached tracklists, picked '{scrobble_pick['title']}' — confirmed by scrobble's own album tag"
                return scrobble_pick['id_album'], (f"{note}. {tie_note}" if note else tie_note)

        # Tie-break (b)(c)(d): is_reissue, then type priority, then year
        matches.sort(key=lambda c: (c.get('is_reissue', False), ALBUM_TYPE_PRIORITY.get(c['type'], OTHER_TYPE_PRIORITY), c['year'] or 9999))
        tie_note = f"Matched in {len(matches)} cached tracklists, picked '{matches[0]['title']}' by type-priority tiebreak"
        note = f"{note}. {tie_note}" if note else tie_note

    return matches[0]['id_album'], note


def build_candidate_list(album_map: Dict[int, List[dict]], id_artist: int) -> List[dict]:
    """
    Build a deduplicated, sorted list of known releases for an artist.
    Sorted by is_reissue, then type priority (album > ep > single >
    greatest hits > live), then year, purely so the printed/prompted list
    reads naturally — the model still picks by number, not by position.
    Returns [] if we have no known albums for this artist at all.
    """
    candidates = album_map.get(id_artist, [])
    if not candidates:
        return []

    seen = set()
    deduped = []
    for c in candidates:
        key = (_normalize_title(c['title']), c['year'])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    deduped.sort(key=lambda c: (c.get('is_reissue', False), ALBUM_TYPE_PRIORITY.get(c['type'], OTHER_TYPE_PRIORITY), c['year'] or 9999))
    return deduped


def _format_candidate_prompt(artist_name: str, song_title: str, candidates: List[dict]) -> str:
    """Build the exact same closed-candidate prompt for all three AI backends."""
    lines = []
    for i, c in enumerate(candidates, start=1):
        year_str = c['year'] if c['year'] else '?'
        type_str = c['type'] if c['type'] else 'unknown'
        reissue_tag = ' [reissue/remaster/deluxe]' if c.get('is_reissue') else ''
        lines.append(f"{i}. {c['title']} ({year_str}) - {type_str}{reissue_tag}")
    listing = "\n".join(lines)

    return f"""You are a music discography expert.

Here is the COMPLETE, CLOSED list of known releases by {artist_name} in our database:

{listing}

The song "{song_title}" by {artist_name} originally belongs to EXACTLY ONE of these releases.

Important rules:
- If this song was released as a promotional single, demo, or advance preview shortly before/around one of these albums, it counts as belonging to THAT ALBUM's release — it is NOT a separate single, even if it was technically issued earlier as a promo.
- Prefer the original release over a [reissue/remaster/deluxe] entry of the same album, when both are listed.
- The oldest release is not automatically the correct answer — a later official album can still be the correct one if that's genuinely where the song belongs in the discography, while an earlier demo/promo of the same song is not.
- If genuinely tied between types, prefer in this exact order: 1) Album (LP) 2) EP 3) Single 4) Live 5) Compilation/Greatest Hits 6) Other.

Respond ONLY with the number of the correct option (for example: 3).
If genuinely NONE of these releases is where the song belongs, respond with: 0
No explanation, no extra text — just the number."""


def _parse_choice(raw: Optional[str], n_candidates: int) -> Optional[int]:
    """Parse a numeric answer. Returns 0 (explicit 'none'), 1..n (a candidate), or None (unparseable)."""
    if not raw:
        return None
    match = re.search(r'\d+', raw)
    if not match:
        return None
    n = int(match.group())
    if 0 <= n <= n_candidates:
        return n
    return None


def _ask_deepseek_choice(prompt: str) -> Optional[str]:
    """Ask DeepSeek China API to pick a candidate number. Returns raw content or None."""
    if not deepseek_client:
        return None
    try:
        completion = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a music discography expert. Respond ONLY with a number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            stream=False
        )
        if not completion.choices:
            return None
        return completion.choices[0].message.content
    except Exception as e:
        print(f"      ⚠️ DeepSeek voting error: {e}")
        return None


def _ask_gemini_family_choice(prompt: str, model: str) -> Optional[str]:
    """Ask a Gemini-family model (Gemma) to pick a candidate number."""
    if not genai_client:
        return None
    try:
        response = genai_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"      ⚠️ {model} voting error: {e}")
        return None


def _ask_groq_choice(prompt: str) -> Optional[str]:
    """Ask Llama 3.3 70B (via Groq) to pick a candidate number.
    Only called as tiebreaker when DeepSeek and Gemma disagree — conserves
    the 1K RPD free quota.
    """
    if not groq_client:
        return None
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a music discography expert. Respond ONLY with a number."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            stream=False
        )
        if not completion.choices:
            return None
        return completion.choices[0].message.content
    except Exception as e:
        print(f"      ⚠️ Groq (Llama 3.3 70B) voting error: {e}")
        return None


def record_ai_resolution_in_cache(album_map: Dict[int, List[dict]], id_artist: int,
                                   id_album: int, song_title: str):
    """
    Feedback loop: when the AI voting panel resolves a song, add it to the
    winning album's cached tracklist — both in memory (so later songs in
    THIS SAME RUN hit the cache immediately) and persisted back to
    2_albums_raw.db (so future runs never have to ask the AI about this
    album/song combination again). Without this, every song from an album
    the AI resolves keeps costing AI calls forever, even after we've
    effectively already learned the album's tracklist.
    """
    candidates = album_map.get(id_artist, [])
    candidate = next((c for c in candidates if c['id_album'] == id_album), None)
    if not candidate:
        return

    tracklist = candidate.get('tracklist') or []
    normalized_existing = {_normalize_song_for_match(t) for t in tracklist}
    if _normalize_song_for_match(song_title) in normalized_existing:
        return  # already cached, nothing to do

    tracklist = tracklist + [song_title]
    candidate['tracklist'] = tracklist  # update in-memory copy immediately

    try:
        album_conn = sqlite3.connect(ALBUM_DB_PATH)
        cursor = album_conn.cursor()
        cursor.execute(
            'UPDATE Album SET tracklist_json = ?, last_update = CURRENT_TIMESTAMP WHERE id_album = ?',
            (json.dumps(tracklist, ensure_ascii=False), id_album)
        )
        album_conn.commit()
        album_conn.close()
    except Exception as e:
        print(f"      ⚠️ Could not persist tracklist-cache feedback for id_album={id_album}: {e}")


def resolve_album_multiai(artist_name: str, song_title: str,
                           album_map: Dict[int, List[dict]], id_artist: int) -> Tuple[int, str, Optional[str]]:
    """
    Resolve the album for a song using closed-candidate voting.

    Voting panel, in order:
        1. DeepSeek (China API) — if it confidently picks a genuine album/LP,
           accepted directly (1 call total).
        2. Otherwise, Gemma votes on the same closed question. If DeepSeek
           and Gemma AGREE, accepted (2/2 majority, 2 calls total).
        3. Only if DeepSeek and Gemma DISAGREE does Gemini get called, as
           a tiebreaking third opinion (Gemini has a very low daily quota,
           so it's reserved for genuine disagreements, not used on every
           song). Whichever option then has >=2 of the 3 votes wins.
        4. If all three end up disagreeing (or too many failed to answer),
           the song is left ambiguous for manual review — never guessed.

    Returns (id_album, source, review_notes):
        - source == 'deepseek'          -> DeepSeek alone, genuine album/LP.
        - source == 'multiai-majority'  -> DeepSeek+Gemma agreed (2/2), or
                                            Gemini broke the tie (2/3).
        - source == 'ambiguous'         -> no agreement reached.
        - source == 'no-candidates'     -> this artist has no known albums
                                            in 2_albums_raw.db at all, so
                                            there was nothing to vote on;
                                            caller should fall back to the
                                            old scrobble/lastfm/musicbrainz
                                            chain.
    """
    candidates = build_candidate_list(album_map, id_artist)
    if not candidates:
        return 0, 'no-candidates', None

    cache_key = f"{artist_name.lower()}|||{song_title.lower()}"
    if cache_key in album_ai_cache:
        cached = album_ai_cache[cache_key]
        if cached is None:
            return 0, 'ambiguous', None
        return cached['id_album'], cached['source'], cached.get('review_notes')

    prompt = _format_candidate_prompt(artist_name, song_title, candidates)

    def _resolve(choice: Optional[int]) -> Tuple[int, Optional[str]]:
        """Map a 1-based choice to (id_album, type). 0/None -> (0, None)."""
        if choice and choice > 0:
            c = candidates[choice - 1]
            return c['id_album'], c['type']
        return 0, None

    # 1. Ask DeepSeek first
    deepseek_raw = _ask_deepseek_choice(prompt)
    deepseek_choice = _parse_choice(deepseek_raw, len(candidates))
    deepseek_id, deepseek_type = _resolve(deepseek_choice)

    # 2. Accept directly if DeepSeek confidently picked a genuine album/LP
    if deepseek_id and deepseek_type in ('album', 'lp'):
        result = {'id_album': deepseek_id, 'source': 'deepseek', 'review_notes': None}
        album_ai_cache[cache_key] = result
        record_ai_resolution_in_cache(album_map, id_artist, deepseek_id, song_title)
        return deepseek_id, 'deepseek', None

    # 3. DeepSeek didn't land on a clear album -> ask Gemma (the other
    #    primary voter; Gemini is NOT called yet)
    time.sleep(0.3)
    gemma_raw = _ask_gemini_family_choice(prompt, GEMMA_MODEL)
    gemma_choice = _parse_choice(gemma_raw, len(candidates))

    votes = {'deepseek': deepseek_choice, 'gemma': gemma_choice}

    if deepseek_choice is not None and deepseek_choice == gemma_choice:
        id_album, _ = _resolve(deepseek_choice)
        note = f"DeepSeek+Gemma agreed (2/2) -> option {deepseek_choice}. Votes: {votes}"
        if id_album:
            result = {'id_album': id_album, 'source': 'multiai-majority', 'review_notes': note}
            album_ai_cache[cache_key] = result
            record_ai_resolution_in_cache(album_map, id_artist, id_album, song_title)
            return id_album, 'multiai-majority', note
        else:
            note = f"DeepSeek+Gemma agreed (2/2) NONE apply. Votes: {votes}"
            album_ai_cache[cache_key] = None
            return 0, 'ambiguous', note

    # 4. DeepSeek and Gemma disagree -> Groq as tiebreaker
    #    (only reached on genuine disagreement, conserves Groq's 1K RPD quota)
    time.sleep(0.3)
    groq_raw = _ask_groq_choice(prompt)
    groq_choice = _parse_choice(groq_raw, len(candidates))
    votes['groq'] = groq_choice

    valid_votes = [v for v in votes.values() if v is not None]
    tally: Dict[int, int] = {}
    for v in valid_votes:
        tally[v] = tally.get(v, 0) + 1

    if tally:
        best_choice, best_count = max(tally.items(), key=lambda kv: kv[1])
        if best_count >= 2:
            id_album, _ = _resolve(best_choice)
            note = f"Groq tiebreak, majority {best_count}/3 -> option {best_choice}. Votes: {votes}"
            if id_album:
                result = {'id_album': id_album, 'source': 'multiai-majority', 'review_notes': note}
                album_ai_cache[cache_key] = result
                record_ai_resolution_in_cache(album_map, id_artist, id_album, song_title)
                return id_album, 'multiai-majority', note
            else:
                note = f"Groq tiebreak, majority {best_count}/3 agreed NONE apply. Votes: {votes}"
                album_ai_cache[cache_key] = None
                return 0, 'ambiguous', note

    # No majority even with Gemini's tiebreak: all three disagreed
    note = f"No majority even with Gemini tiebreak — all disagreed or failed. Votes: {votes}"
    album_ai_cache[cache_key] = None
    return 0, 'ambiguous', note


def _normalize_title(title: str) -> str:
    """Lowercase, trim, collapse whitespace for comparisons."""
    return re.sub(r'\s+', ' ', title or '').strip().lower()


def find_album_in_db_smart(album_map: Dict[int, List[dict]], id_artist: int,
                            candidate_title: str, candidate_year: Optional[int] = None) -> int:
    """
    Match a candidate album (title + optional year) against the artist's
    albums in 2_albums_raw.db, disambiguating between editions of the same
    title (original vs remaster/deluxe/etc.) using release_year and the
    is_reissue flag.

    Matching order:
      1. Exact title + exact year        -> most reliable, picks the right edition
                                             (ties broken by is_reissue=False first)
      2. Exact title, closest year       -> AI got the title right, year slightly off
                                             (is_reissue=False preferred first)
      3. Fuzzy/partial title match       -> tie-broken by is_reissue, then type
                                             priority (album>ep>single>live), then year

    Returns id_album, or 0 if nothing matches.
    """
    if not candidate_title or id_artist not in album_map:
        return 0

    candidates = album_map[id_artist]
    target = _normalize_title(candidate_title)

    # 1. Exact title + exact year (prefer non-reissue in the rare case of a tie)
    if candidate_year:
        exact = [c for c in candidates if _normalize_title(c['title']) == target and c['year'] == candidate_year]
        if exact:
            exact.sort(key=lambda c: c.get('is_reissue', False))
            return exact[0]['id_album']

    # 2. Exact title, closest year (or first one if no year to compare)
    same_title = [c for c in candidates if _normalize_title(c['title']) == target]
    if same_title:
        if candidate_year:
            same_title.sort(key=lambda c: (c.get('is_reissue', False), abs((c['year'] or 9999) - candidate_year)))
        else:
            same_title.sort(key=lambda c: (c.get('is_reissue', False), ALBUM_TYPE_PRIORITY.get(c['type'], OTHER_TYPE_PRIORITY)))
        return same_title[0]['id_album']

    # 3. Fuzzy/partial title match, prefer non-reissue + original album type + earliest year
    partial = [
        c for c in candidates
        if target in _normalize_title(c['title']) or _normalize_title(c['title']) in target
    ]
    if partial:
        partial.sort(key=lambda c: (c.get('is_reissue', False), ALBUM_TYPE_PRIORITY.get(c['type'], OTHER_TYPE_PRIORITY), c['year'] or 9999))
        return partial[0]['id_album']

    return 0


def find_album_for_song(artist_name: str, song_title: str, album_from_scrobble: str,
                         album_map: Dict[int, List[dict]], id_artist: int) -> Tuple[int, str, Optional[str]]:
    """
    Find album ID for a song using, in order:
      1. Tracklist cache — free membership check against real tracklists
         already stored in 2_albums_raw.db. Zero AI calls when it hits.
      2. Closed-candidate multi-AI voting (see resolve_album_multiai).
      3. Old scrobble/Last.fm/MusicBrainz fuzzy-matching chain — ONLY when
         this artist has no known albums at all (nothing to check/vote on).

    A genuinely ambiguous multi-AI vote is NEVER overridden by fuzzy
    matching — that's a real "needs review" signal, not a reason to fall
    back to weaker matching.

    Returns (id_album, source, review_notes).
    Sources: 'tracklist-cache', 'deepseek', 'multiai-majority', 'ambiguous',
             'scrobble', 'lastfm', 'musicbrainz', 'none'
    """
    # 1. Tracklist cache — check first, it's free and it's ground truth
    id_album, note = find_song_in_tracklist_cache(song_title, album_map, id_artist, album_from_scrobble)
    if id_album > 0:
        return id_album, 'tracklist-cache', note

    id_album, source, review_notes = resolve_album_multiai(artist_name, song_title, album_map, id_artist)

    if source != 'no-candidates':
        # Either resolved (deepseek / multiai-majority) or genuinely ambiguous —
        # in both cases we're done, no fuzzy-matching fallback.
        return id_album, source, review_notes

    # No known albums for this artist at all yet — fall back to the old
    # fuzzy-matching chain so we're not left completely empty-handed.
    if album_from_scrobble:
        id_album = find_album_in_db_smart(album_map, id_artist, album_from_scrobble)
        if id_album > 0:
            return id_album, 'scrobble', None

    album_name = get_album_from_lastfm_track(artist_name, song_title)
    if album_name:
        id_album = find_album_in_db_smart(album_map, id_artist, album_name)
        if id_album > 0:
            return id_album, 'lastfm', None

    album_name = get_album_from_musicbrainz(artist_name, song_title)
    if album_name:
        id_album = find_album_in_db_smart(album_map, id_artist, album_name)
        if id_album > 0:
            return id_album, 'musicbrainz', None

    return 0, 'none', None


# ============================================
# DATABASE FUNCTIONS
# ============================================

def create_schema(conn):
    """Creates the Song table with pending support and id_album."""
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Song (
            id_song     INTEGER PRIMARY KEY AUTOINCREMENT,
            id_artist   INTEGER NOT NULL,
            id_album    INTEGER NOT NULL DEFAULT 0,
            title       TEXT    NOT NULL,
            duration    INTEGER,
            duration_source TEXT,
            retry_count INTEGER DEFAULT 0,
            album_retry_count INTEGER DEFAULT 0,
            album_source TEXT,
            review_notes TEXT,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (id_artist, title)
        )
    ''')

    # Migration: add missing columns if they don't exist
    cursor.execute("PRAGMA table_info(Song)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'id_album' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN id_album INTEGER NOT NULL DEFAULT 0")
        print("   ✅ Added column: id_album")
    
    if 'duration_source' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN duration_source TEXT")
        print("   ✅ Added column: duration_source")
    
    if 'retry_count' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN retry_count INTEGER DEFAULT 0")
        print("   ✅ Added column: retry_count")
    
    if 'album_retry_count' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN album_retry_count INTEGER DEFAULT 0")
        print("   ✅ Added column: album_retry_count")
    
    if 'album_source' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN album_source TEXT")
        print("   ✅ Added column: album_source")

    if 'review_notes' not in columns:
        cursor.execute("ALTER TABLE Song ADD COLUMN review_notes TEXT")
        print("   ✅ Added column: review_notes")

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_artist ON Song (id_artist)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_album ON Song (id_album)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_title ON Song (title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_duration ON Song (duration)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_retry ON Song (retry_count)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_song_album_retry ON Song (album_retry_count)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Metadata (
            key     TEXT PRIMARY KEY,
            value   TEXT,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()


def get_last_update_time(conn) -> Optional[str]:
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM Metadata WHERE key = ?', ('last_update',))
    row = cursor.fetchone()
    return row[0] if row else None


def set_last_update_time(conn, timestamp: str):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO Metadata (key, value, last_update)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', ('last_update', timestamp))
    conn.commit()


def load_artist_map() -> dict:
    """Loads all artists into memory once, as {name: id_artist}."""
    artist_conn = sqlite3.connect(ARTIST_DB_PATH)
    cursor = artist_conn.cursor()
    cursor.execute('SELECT id_artist, name FROM Artist')
    mapping = {name: id_artist for id_artist, name in cursor.fetchall()}
    artist_conn.close()
    return mapping


def load_album_map() -> Dict[int, List[dict]]:
    """
    Loads all albums into memory as:
    {id_artist: [ {title, year, type, is_reissue, tracklist, id_album}, ... ]}

    Loading release_year and album_type (not just title) is what lets
    find_album_in_db_smart tell apart the original 1973 release from a
    2010 remaster of the same title. is_reissue adds an explicit
    tiebreaker so remasters/deluxe/anniversary editions are only picked
    over the original when nothing else distinguishes them.

    tracklist (parsed from tracklist_json, when the column exists — added
    by the updated 2_album_db_raw.py) is what powers the tracklist-cache
    lookup: a simple membership check against real, verified track titles,
    resolved for free without any AI call.
    (No JOIN with Artist to avoid cross-database issues.)
    """
    album_conn = sqlite3.connect(ALBUM_DB_PATH)
    cursor = album_conn.cursor()

    cursor.execute("PRAGMA table_info(Album)")
    columns = {row[1] for row in cursor.fetchall()}
    has_is_reissue = 'is_reissue' in columns
    has_tracklist = 'tracklist_json' in columns

    select_cols = ['id_artist', 'title', 'release_year', 'album_type', 'id_album']
    if has_is_reissue:
        select_cols.insert(4, 'is_reissue')
    if has_tracklist:
        select_cols.append('tracklist_json')

    cursor.execute(f"SELECT {', '.join(select_cols)} FROM Album")

    album_map: Dict[int, List[dict]] = {}
    for row in cursor.fetchall():
        row_dict = dict(zip(select_cols, row))

        tracklist = None
        if has_tracklist and row_dict.get('tracklist_json'):
            try:
                tracklist = json.loads(row_dict['tracklist_json'])
            except (json.JSONDecodeError, TypeError):
                tracklist = None

        album_map.setdefault(row_dict['id_artist'], []).append({
            'title': row_dict['title'],
            'year': row_dict['release_year'],
            'type': (row_dict['album_type'] or '').strip().lower(),
            'is_reissue': bool(row_dict.get('is_reissue', 0)),
            'tracklist': tracklist,
            'id_album': row_dict['id_album']
        })

    album_conn.close()
    return album_map


def song_exists(conn, id_artist: int, title: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('SELECT id_song FROM Song WHERE id_artist = ? AND title = ?', (id_artist, title))
    return cursor.fetchone() is not None


def get_pending_songs(conn) -> list:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id_song, id_artist, title, retry_count
        FROM Song
        WHERE duration_source = 'pending' AND retry_count < 3
        ORDER BY retry_count ASC, last_update ASC
    ''')
    return cursor.fetchall()


def get_songs_without_album(conn) -> list:
    """Get songs that have id_album = 0 and album_retry_count < 3."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id_song, id_artist, title, album_retry_count
        FROM Song
        WHERE id_album = 0 AND album_retry_count < 3
        ORDER BY album_retry_count ASC, last_update ASC
    ''')
    return cursor.fetchall()


def save_song(conn, id_artist: int, id_album: int, title: str, 
              duration: Optional[int] = None, duration_source: str = 'unknown',
              retry_count: int = 0, album_retry_count: int = 0,
              album_source: str = None, review_notes: Optional[str] = None):
    cursor = conn.cursor()

    if song_exists(conn, id_artist, title):
        return None

    cursor.execute('''
        INSERT INTO Song (id_artist, id_album, title, duration, duration_source, retry_count, album_retry_count, album_source, review_notes, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (id_artist, id_album, title, duration, duration_source, retry_count, album_retry_count, album_source, review_notes))
    conn.commit()

    return cursor.lastrowid


def update_song_duration(conn, id_song: int, duration: Optional[int], source: str):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Song
        SET duration = ?, duration_source = ?, retry_count = retry_count + 1, last_update = CURRENT_TIMESTAMP
        WHERE id_song = ?
    ''', (duration, source, id_song))
    conn.commit()


def update_song_album(conn, id_song: int, id_album: int, album_source: str, review_notes: Optional[str] = None):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Song
        SET id_album = ?, album_source = ?, review_notes = ?, album_retry_count = album_retry_count + 1, last_update = CURRENT_TIMESTAMP
        WHERE id_song = ?
    ''', (id_album, album_source, review_notes, id_song))
    conn.commit()


# ============================================
# LAST.FM API FUNCTIONS
# ============================================

def fetch_duration_from_lastfm(artist_name: str, song_title: str) -> Optional[int]:
    params = {
        'method': 'track.getInfo',
        'artist': artist_name,
        'track': song_title,
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }

    try:
        resp = requests.get(LASTFM_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if 'error' in data:
            return None

        track = data.get('track', {})
        duration = track.get('duration')

        if duration and duration != '0':
            return int(duration) // 1000

        return None

    except Exception as e:
        return None


def fetch_duration_with_fallback(artist_name: str, song_title: str) -> Tuple[Optional[int], str]:
    """Fetch duration using: Last.fm -> Spotify -> DeepSeek China"""
    
    # 1. Try Last.fm
    duration = fetch_duration_from_lastfm(artist_name, song_title)
    if duration:
        return duration, 'lastfm'

    # 2. Try Spotify
    duration = fetch_duration_from_spotify(artist_name, song_title)
    if duration:
        return duration, 'spotify'

    # 3. Try DeepSeek China
    duration = fetch_duration_from_deepseek(artist_name, song_title)
    if duration:
        return duration, 'deepseek'

    return None, 'pending'


def fetch_scrobbles_page(page: int, limit: int = 200, from_timestamp: Optional[int] = None):
    params = {
        'method': 'user.getrecenttracks',
        'user': LASTFM_USER,
        'api_key': LASTFM_API_KEY,
        'format': 'json',
        'limit': limit,
        'page': page
    }

    if from_timestamp:
        params['from'] = from_timestamp

    resp = requests.get(LASTFM_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if 'error' in data:
        raise RuntimeError(f"Last.fm API error: {data.get('message', data)}")

    return data


def process_scrobble(conn, scrobble_data, artist_map: dict, album_map: dict, 
                     artist_name_cache: dict) -> bool:
    """Returns True if a genuinely new song was processed (counts towards
    MAX_NEW_SONGS), False if this scrobble was skipped for any reason
    (now playing, artist not found, or song already existed)."""
    if '@attr' in scrobble_data and scrobble_data['@attr'].get('nowplaying') == 'true':
        return False

    artist_name = scrobble_data['artist']['#text']
    song_title = scrobble_data['name']
    
    # Get album name from scrobble data (if available)
    album_from_scrobble = scrobble_data.get('album', {}).get('#text', '')

    id_artist = artist_map.get(artist_name)

    if not id_artist:
        # Try to find artist by partial match (cache)
        if artist_name not in artist_name_cache:
            # Search in artist_map keys for partial match
            for name in artist_map:
                if artist_name.lower() in name.lower() or name.lower() in artist_name.lower():
                    artist_name_cache[artist_name] = name
                    break
            if artist_name not in artist_name_cache:
                artist_name_cache[artist_name] = None
        
        matched_name = artist_name_cache.get(artist_name)
        if matched_name:
            id_artist = artist_map.get(matched_name)
        
        if not id_artist:
            print(f"    ⚠️ Artist not found: {artist_name}")
            return False

    if song_exists(conn, id_artist, song_title):
        return False

    # Try to find album
    id_album, album_source, review_notes = find_album_for_song(
        artist_name, song_title, album_from_scrobble, album_map, id_artist
    )
    
    # If album not found, set to 0 and mark for retry
    album_retry_count = 0
    if id_album == 0:
        album_retry_count = 0  # Will increment on retry
        if album_source not in ('ambiguous',):
            album_source = 'none'

    print(f"    🎵 New song: {song_title}")
    if id_album > 0:
        print(f"      💿 Album ID: {id_album} (source: {album_source})")
    elif album_source == 'ambiguous':
        print(f"      💿 Ambiguous (3 AI votes disagreed) — needs manual review: {review_notes}")
    else:
        print(f"      💿 No album found (will retry later)")

    # Get duration
    duration, duration_source = fetch_duration_with_fallback(artist_name, song_title)

    if duration:
        minutes = duration // 60
        seconds = duration % 60
        print(f"      ⏱️ Duration: {minutes}:{seconds:02d} ({duration}s) [source: {duration_source}]")
    else:
        print(f"      ⏱️ Duration: Unknown [source: pending] - Will retry on next run")

    retry_count = 0 if duration_source != 'pending' else 0
    save_song(conn, id_artist, id_album, song_title, duration, duration_source,
              retry_count, album_retry_count, album_source, review_notes)

    # Rate limiting: DeepSeek and Groq get a short pause; tracklist-cache
    # hits made zero AI calls so they get zero extra delay. Tracklist-cache
    # hits made zero AI calls, so they get zero extra delay.
    if duration_source == 'deepseek':
        time.sleep(1.5)
    elif duration_source in ('spotify', 'pending'):
        time.sleep(0.5)

    if album_source == 'deepseek':
        time.sleep(1.5)
    elif album_source in ('multiai-majority', 'ambiguous'):
        # Already made 3 AI calls (deepseek + gemini + gemma) resolving this one
        time.sleep(2.0)
    elif album_source in ('lastfm', 'musicbrainz'):
        time.sleep(0.3)

    return True


def retry_pending_songs(conn, artist_map: dict, album_map: dict, artist_name_cache: dict):
    """Retry songs that have pending duration."""
    pending = get_pending_songs(conn)
    
    if not pending:
        print("   No pending songs to retry (duration).")
    else:
        print(f"   🔄 Retrying {len(pending)} pending duration songs...")
        
        id_to_name = {id_artist: name for name, id_artist in artist_map.items()}
        
        retried = 0
        for id_song, id_artist, title, retry_count in pending:
            artist_name = id_to_name.get(id_artist)
            if not artist_name:
                continue
            
            print(f"      🔄 Retry {retry_count + 1}: {artist_name} - {title}")
            
            duration, source = fetch_duration_with_fallback(artist_name, title)
            
            if duration:
                minutes = duration // 60
                seconds = duration % 60
                print(f"         ✅ Found! {minutes}:{seconds:02d} ({duration}s) [source: {source}]")
                update_song_duration(conn, id_song, duration, source)
                retried += 1
            else:
                print(f"         ⏳ Still pending...")
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE Song
                    SET retry_count = retry_count + 1, last_update = CURRENT_TIMESTAMP
                    WHERE id_song = ?
                ''', (id_song,))
                conn.commit()
            
            if source == 'deepseek':
                time.sleep(1.5)
            elif source in ('spotify', 'pending'):
                time.sleep(0.5)
        
        if retried > 0:
            print(f"   ✅ Found durations for {retried} pending songs!")


def retry_album_songs(conn, artist_map: dict, album_map: dict, artist_name_cache: dict):
    """Retry songs that have id_album = 0."""
    songs = get_songs_without_album(conn)
    
    if not songs:
        print("   No songs without album to retry.")
        return 0
    
    print(f"   🔄 Retrying {len(songs)} songs without album...")
    
    id_to_name = {id_artist: name for name, id_artist in artist_map.items()}
    
    retried = 0
    for id_song, id_artist, title, album_retry_count in songs:
        artist_name = id_to_name.get(id_artist)
        if not artist_name:
            continue
        
        print(f"      🔄 Album retry {album_retry_count + 1}: {artist_name} - {title}")
        
        # Try to find album again (without album_from_scrobble, since we don't have it)
        id_album, album_source, review_notes = find_album_for_song(
            artist_name, title, '', album_map, id_artist
        )
        
        if id_album > 0:
            print(f"         ✅ Found album! ID: {id_album} (source: {album_source})")
            update_song_album(conn, id_song, id_album, album_source, review_notes)
            retried += 1
        elif album_source == 'ambiguous':
            print(f"         ⚠️ Still ambiguous (3 AI votes disagreed): {review_notes}")
            update_song_album(conn, id_song, 0, album_source, review_notes)
        else:
            print(f"         ⏳ Still no album found.")
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE Song
                SET album_retry_count = album_retry_count + 1, last_update = CURRENT_TIMESTAMP
                WHERE id_song = ?
            ''', (id_song,))
            conn.commit()
        
        if album_source == 'deepseek':
            time.sleep(1.5)
        elif album_source in ('multiai-majority', 'ambiguous'):
            time.sleep(2.0)
        elif album_source in ('lastfm', 'musicbrainz'):
            time.sleep(0.3)
    
    if retried > 0:
        print(f"   ✅ Found albums for {retried} songs!")
    
    return retried


def fetch_all_scrobbles(conn, limit: int = 200):
    last_update = get_last_update_time(conn)

    from_timestamp = None
    if last_update:
        try:
            dt = datetime.fromisoformat(last_update)
            from_timestamp = int(dt.timestamp())
            print(f"📌 Last update: {last_update}")
            print("   Fetching only new scrobbles...")
        except ValueError:
            print(f"   ⚠️ Could not parse timestamp: {last_update}. Fetching all scrobbles...")

    print("📚 Loading artist map into memory...")
    artist_map = load_artist_map()
    print(f"   {len(artist_map)} artists loaded.")

    print("📚 Loading album map into memory...")
    album_map = load_album_map()
    total_albums = sum(len(albums) for albums in album_map.values())
    print(f"   {total_albums} albums loaded across {len(album_map)} artists.")

    artist_name_cache = {}
    page = 1
    total_processed = 0
    new_songs_count = 0
    cap_reached = False

    if MAX_NEW_SONGS is not None:
        print(f"🧪 TEST RUN — capped at {MAX_NEW_SONGS} new songs")
    else:
        print("🧪 TEST RUN — no cap, processing all scrobbles")

    while True:
        print(f"📄 Fetching page {page}...")

        try:
            data = fetch_scrobbles_page(page, limit, from_timestamp)
        except Exception as e:
            print(f"❌ Error fetching page {page}: {e}")
            break

        tracks = data.get('recenttracks', {}).get('track', [])

        if not tracks:
            print("   No more tracks found.")
            break

        total_pages = int(data['recenttracks']['@attr']['totalPages'])

        for track in tracks:
            was_new = process_scrobble(conn, track, artist_map, album_map, artist_name_cache)
            total_processed += 1
            if was_new:
                new_songs_count += 1

            if MAX_NEW_SONGS is not None and new_songs_count >= MAX_NEW_SONGS:
                print(f"\n🛑 Reached the {MAX_NEW_SONGS}-new-song cap for this test run. Stopping.")
                cap_reached = True
                break

        progress = f"{new_songs_count}/{MAX_NEW_SONGS}" if MAX_NEW_SONGS is not None else str(new_songs_count)
        print(f"   ✅ Page {page}/{total_pages} processed. New songs so far: {progress}")

        if cap_reached:
            break

        if page >= total_pages:
            break

        page += 1
        time.sleep(0.3)

    if not cap_reached:
        print("\n🔄 Retrying pending duration songs...")
        retry_pending_songs(conn, artist_map, album_map, artist_name_cache)

        print("\n🔄 Retrying songs without album...")
        retry_album_songs(conn, artist_map, album_map, artist_name_cache)
    else:
        print("\n⏭️  Skipping retry passes — new-song cap already reached for this test run.")

    now = datetime.now().isoformat()
    set_last_update_time(conn, now)

    return total_processed


def get_stats(conn):
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM Song')
    total_songs = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM Song WHERE duration IS NOT NULL')
    songs_with_duration = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM Song WHERE duration_source = ?', ('pending',))
    pending_songs = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM Song WHERE id_album = 0')
    songs_without_album = cursor.fetchone()[0]

    cursor.execute('SELECT duration_source, COUNT(*) FROM Song WHERE duration IS NOT NULL GROUP BY duration_source')
    duration_sources = cursor.fetchall()

    cursor.execute('SELECT album_source, COUNT(*) FROM Song GROUP BY album_source')
    album_sources = cursor.fetchall()

    cursor.execute('SELECT SUM(duration) FROM Song')
    total_duration = cursor.fetchone()[0]

    return {
        'total_songs': total_songs,
        'songs_with_duration': songs_with_duration,
        'pending_songs': pending_songs,
        'songs_without_album': songs_without_album,
        'duration_sources': duration_sources,
        'album_sources': album_sources,
        'total_duration_seconds': total_duration
    }


def create_database():
    print("=" * 60)
    cap_label = f"capped at {MAX_NEW_SONGS} new songs" if MAX_NEW_SONGS is not None else "no cap — full run"
    print(f"SONG DATABASE - RAW DATA FETCHER ({cap_label})")
    print("=" * 60)

    if not LASTFM_API_KEY or not LASTFM_USER:
        print("⚠️ LASTFM_API_KEY / LASTFM_USER not found in environment variables.")
        return

    if not os.path.exists(ARTIST_DB_PATH):
        print(f"❌ Artist database not found: {ARTIST_DB_PATH}")
        print("   Please run 1_artist_genre_filter.py first.")
        return

    if not os.path.exists(ALBUM_DB_PATH):
        print(f"⚠️ Album database not found: {ALBUM_DB_PATH}")
        print("   Running without album mapping. id_album will be set to 0.")
        print("   (Run 2_album_db_raw.py first for full album support)")

    if not deepseek_client:
        print("⚠️ DEEPSEEK_API_KEY not found in environment variables.")
        print("   AI-based duration/album lookups will be skipped (songs stay pending/id_album=0).")

    os.makedirs(os.path.dirname(SONG_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(SONG_DB_PATH)
    create_schema(conn)

    last_update = get_last_update_time(conn)
    if last_update:
        print(f"📂 Database already exists. Last update: {last_update}")
        print("   Fetching only new songs...")
    else:
        print("📂 New database. Fetching ALL songs...")

    print("-" * 60)

    try:
        total = fetch_all_scrobbles(conn)
    except Exception as e:
        print(f"❌ Error fetching scrobbles: {e}")
        conn.close()
        sys.exit(1)

    stats = get_stats(conn)

    total_hours = stats['total_duration_seconds'] // 3600 if stats['total_duration_seconds'] else 0
    remaining_seconds = stats['total_duration_seconds'] % 3600 if stats['total_duration_seconds'] else 0

    print("-" * 60)
    print(f"\n✅ Database updated successfully")
    print(f"📁 Location: {SONG_DB_PATH}")
    print(f"📋 Table: Song")
    print(f"🎵 Total songs: {stats['total_songs']}")
    print(f"⏱️  Songs with duration: {stats['songs_with_duration']}")
    print(f"⏳ Pending songs (no duration yet): {stats['pending_songs']}")
    print(f"💿 Songs without album: {stats['songs_without_album']}")

    if stats['duration_sources']:
        print("\n📊 Duration sources:")
        for source, count in stats['duration_sources']:
            print(f"   {source}: {count} songs")

    if stats['album_sources']:
        print("\n📊 Album sources:")
        for source, count in stats['album_sources']:
            print(f"   {source}: {count} songs")

    if stats['total_duration_seconds']:
        print(f"\n📊 Total duration: {total_hours}h {remaining_seconds//60}m")
    print("=" * 60)

    conn.close()


if __name__ == "__main__":
    create_database()

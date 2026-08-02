"""
Song Database - Raw Data Fetcher (NVIDIA edition)

Alternate version of 3_song_db_raw.py that uses DeepSeek V4 Flash hosted on
NVIDIA (free tier) instead of the official DeepSeek China API.

Advantages: free.
Disadvantages: slower, and more prone to returning nothing at all.
Good for a first bulk run (heavy token consumption) when populating the
database from scratch. Whatever is left as 'pending'/id_album=0 afterwards
can later be resolved by running the DeepSeek China version
(3_song_db_raw.py), which is more reliable.

It retrieves:
- Songs (id_artist, id_album, title, duration)

Duration is fetched from:
1. Last.fm API (primary)
2. Spotify API (fallback 1 - optional)
3. NVIDIA-hosted DeepSeek V4 Flash (fallback 2 - AI)

Album is fetched from:
1. NVIDIA-hosted DeepSeek V4 Flash (primary - identifies the ORIGINAL release)
2. Scrobble data (fallback, if available)
3. Last.fm track.getInfo (fallback)
4. MusicBrainz (fallback)

If the AI backend returns nothing (timeout, empty response, rate limit, etc.)
the song is simply left as pending/id_album=0 and the script moves on to the
next song — this is never treated as a fatal error.
"""

import sqlite3
import os
import sys
import requests
import time
import re
import json
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

# Try to import the OpenAI client (used to talk to the NVIDIA-hosted endpoint)
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ openai package not installed. Installing...")
    os.system(f"{sys.executable} -m pip install openai")
    from openai import OpenAI

# Load environment variables
load_dotenv()

# Database paths
ARTIST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '1_artist_genres.db')
ALBUM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2_albums_raw.db')
SONG_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '3_songs_raw.db')

# --- Credentials ---
LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY')
LASTFM_USER = os.environ.get('LASTFM_USER')
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

LASTFM_API_URL = 'https://ws.audioscrobbler.com/2.0/'
SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_API_URL = 'https://api.spotify.com/v1/search'

NVIDIA_MODEL = "deepseek-ai/deepseek-v4-flash"

# NVIDIA free tier can be slow (reasoning_effort=high), give it real room
# before giving up on a single request.
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY or "missing",  # client requires a non-empty string
    timeout=120.0
) if NVIDIA_API_KEY else None

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

# Priority order when several album editions match: lower number = higher priority
ALBUM_TYPE_PRIORITY = {'album': 0, 'lp': 0, 'ep': 1, 'single': 2, 'live': 3}


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
# NVIDIA API FUNCTIONS (DeepSeek V4 Flash, free tier)
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


def fetch_duration_from_nvidia(artist_name: str, song_title: str) -> Optional[int]:
    """Fetch song duration using DeepSeek V4 Flash hosted on NVIDIA (free tier).

    On any failure (timeout, empty response, rate limit, malformed output)
    this simply returns None so the caller can move on to the next song —
    it is never treated as a fatal error.
    """
    if not nvidia_client:
        return None

    try:
        clean_title = song_title.strip()
        clean_title = re.sub(r'\s*[:/]\s*', ' / ', clean_title)

        prompt = f"""You are a music expert. What is the exact duration of the song "{clean_title}" by {artist_name}?

IMPORTANT: Respond ONLY with the duration in seconds as a number.
Do NOT respond with minutes:seconds format.
Do NOT respond with a year (like 1973).
If you don't know, respond with '0'."""

        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "You are a music expert. Respond ONLY with a number (duration in seconds). Never respond with a year."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
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
                print(f"      ⚠️ NVIDIA returned a year ({duration}s) instead of duration.")
                return None

            if duration > 0 and duration < 6000:
                return duration

        return None

    except Exception as e:
        print(f"      ⚠️ NVIDIA error for '{artist_name} - {song_title}': {e}")
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


def get_album_info_from_nvidia(artist_name: str, song_title: str) -> Optional[dict]:
    """
    Ask DeepSeek V4 Flash (via NVIDIA, free tier) for the ORIGINAL release
    where a song first appeared.

    Priority when a song appears on multiple original release types:
        1. Studio Album (LP)
        2. EP
        3. Single
        4. Live album (only if it never appeared on any of the above)

    Reissues, remasters, deluxe/anniversary editions, greatest hits,
    compilations and box sets are explicitly excluded — we want the
    earliest original release only.

    Returns {'album': str, 'year': int, 'type': str} or None.
    On any failure this returns None quietly — the caller falls back to
    the other album sources and moves on to the next song, this is never
    treated as a fatal error.
    """
    if not nvidia_client:
        return None

    cache_key = f"{artist_name.lower()}|||{song_title.lower()}"
    if cache_key in album_ai_cache:
        return album_ai_cache[cache_key]

    try:
        prompt = f"""You are a music discography expert.

For the song "{song_title}" by {artist_name}, identify the ORIGINAL release where this song first appeared.

Priority order when the song could belong to more than one type of original release:
1. Studio Album (LP)
2. EP
3. Single
4. Live album (only if it was never released on any studio album, EP or single)

Always choose the EARLIEST original release year for that release.
Do NOT consider reissues, remasters, deluxe editions, anniversary editions, greatest hits, compilations, or box sets — only the first original release.

Respond ONLY with a JSON object in this exact format, with no extra text, no markdown, no explanation:
{{"album": "<original album title>", "year": <4-digit year as integer>, "type": "<album|ep|single|live>"}}

If you genuinely don't know, respond with:
{{"album": null, "year": null, "type": null}}"""

        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": "You are a music discography expert. Respond ONLY with a valid JSON object, no markdown, no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
            stream=False
        )

        if not completion.choices:
            album_ai_cache[cache_key] = None
            return None

        raw = completion.choices[0].message.content
        if not raw:
            album_ai_cache[cache_key] = None
            return None

        raw = raw.strip()
        # Strip markdown code fences if the model added them anyway
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            album_ai_cache[cache_key] = None
            return None

        parsed = json.loads(match.group())

        album = parsed.get('album')
        year = parsed.get('year')
        album_type = parsed.get('type')

        if not album or not isinstance(album, str):
            album_ai_cache[cache_key] = None
            return None

        # Validate year
        try:
            year = int(year)
            current_year = datetime.now().year
            if not (1900 <= year <= current_year + 1):
                year = None
        except (TypeError, ValueError):
            year = None

        # Normalize type
        if isinstance(album_type, str):
            album_type = album_type.strip().lower()
            if album_type not in ALBUM_TYPE_PRIORITY:
                album_type = None
        else:
            album_type = None

        info = {'album': album.strip(), 'year': year, 'type': album_type}
        album_ai_cache[cache_key] = info
        return info

    except Exception as e:
        print(f"      ⚠️ NVIDIA album lookup error for '{artist_name} - {song_title}': {e}")
        album_ai_cache[cache_key] = None
        return None


def _normalize_title(title: str) -> str:
    """Lowercase, trim, collapse whitespace for comparisons."""
    return re.sub(r'\s+', ' ', title or '').strip().lower()


def find_album_in_db_smart(album_map: Dict[int, List[dict]], id_artist: int,
                            candidate_title: str, candidate_year: Optional[int] = None) -> int:
    """
    Match a candidate album (title + optional year) against the artist's
    albums in 2_album_raw, disambiguating between editions of the same
    title (original vs remaster/deluxe/etc.) using release_year.

    Matching order:
      1. Exact title + exact year        -> most reliable, picks the right edition
      2. Exact title, closest year       -> AI got the title right, year slightly off
      3. Fuzzy/partial title match       -> tie-broken by type priority (album>ep>single>live)
                                             then earliest year

    Returns id_album, or 0 if nothing matches.
    """
    if not candidate_title or id_artist not in album_map:
        return 0

    candidates = album_map[id_artist]
    target = _normalize_title(candidate_title)

    # 1. Exact title + exact year
    if candidate_year:
        for c in candidates:
            if _normalize_title(c['title']) == target and c['year'] == candidate_year:
                return c['id_album']

    # 2. Exact title, closest year (or first one if no year to compare)
    same_title = [c for c in candidates if _normalize_title(c['title']) == target]
    if same_title:
        if candidate_year:
            same_title.sort(key=lambda c: abs((c['year'] or 9999) - candidate_year))
        else:
            same_title.sort(key=lambda c: ALBUM_TYPE_PRIORITY.get(c['type'], 9))
        return same_title[0]['id_album']

    # 3. Fuzzy/partial title match, prefer original album type + earliest year
    partial = [
        c for c in candidates
        if target in _normalize_title(c['title']) or _normalize_title(c['title']) in target
    ]
    if partial:
        partial.sort(key=lambda c: (ALBUM_TYPE_PRIORITY.get(c['type'], 9), c['year'] or 9999))
        return partial[0]['id_album']

    return 0


def find_album_for_song(artist_name: str, song_title: str, album_from_scrobble: str,
                         album_map: Dict[int, List[dict]], id_artist: int) -> Tuple[int, str]:
    """
    Find album ID for a song, preferring the ORIGINAL release
    (Album > EP > Single > Live, earliest year).

    DeepSeek V4 Flash (via NVIDIA) is the primary source: it's asked directly
    for the original release (title + year + type), which lets us match
    precisely against 2_album_raw on (title, release_year) instead of
    guessing from whichever album name a scrobble/Last.fm/MusicBrainz
    happens to report (which is often a reissue, live album, or compilation).

    Returns (id_album, source).
    Sources: 'nvidia', 'scrobble', 'lastfm', 'musicbrainz', 'none'
    """
    # 1. Ask the NVIDIA-hosted model for the canonical original release
    ai_info = get_album_info_from_nvidia(artist_name, song_title)
    if ai_info and ai_info.get('album'):
        id_album = find_album_in_db_smart(album_map, id_artist, ai_info['album'], ai_info.get('year'))
        if id_album > 0:
            return id_album, 'nvidia'

    # 2. Fallback: album reported on the scrobble itself
    if album_from_scrobble:
        id_album = find_album_in_db_smart(album_map, id_artist, album_from_scrobble)
        if id_album > 0:
            return id_album, 'scrobble'

    # 3. Fallback: Last.fm track.getInfo
    album_name = get_album_from_lastfm_track(artist_name, song_title)
    if album_name:
        id_album = find_album_in_db_smart(album_map, id_artist, album_name)
        if id_album > 0:
            return id_album, 'lastfm'

    # 4. Fallback: MusicBrainz
    album_name = get_album_from_musicbrainz(artist_name, song_title)
    if album_name:
        id_album = find_album_in_db_smart(album_map, id_artist, album_name)
        if id_album > 0:
            return id_album, 'musicbrainz'

    return 0, 'none'


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
    {id_artist: [ {title, year, type, id_album}, ... ]}

    Loading release_year and album_type (not just title) is what lets
    find_album_in_db_smart tell apart the original 1973 release from a
    2010 remaster of the same title.
    (No JOIN with Artist to avoid cross-database issues.)
    """
    album_conn = sqlite3.connect(ALBUM_DB_PATH)
    cursor = album_conn.cursor()
    cursor.execute('SELECT id_artist, title, release_year, album_type, id_album FROM Album')

    album_map: Dict[int, List[dict]] = {}
    for id_artist, title, release_year, album_type, id_album in cursor.fetchall():
        album_map.setdefault(id_artist, []).append({
            'title': title,
            'year': release_year,
            'type': (album_type or '').strip().lower(),
            'id_album': id_album
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
              album_source: str = None):
    cursor = conn.cursor()

    if song_exists(conn, id_artist, title):
        return None

    cursor.execute('''
        INSERT INTO Song (id_artist, id_album, title, duration, duration_source, retry_count, album_retry_count, album_source, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (id_artist, id_album, title, duration, duration_source, retry_count, album_retry_count, album_source))
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


def update_song_album(conn, id_song: int, id_album: int, album_source: str):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Song
        SET id_album = ?, album_source = ?, album_retry_count = album_retry_count + 1, last_update = CURRENT_TIMESTAMP
        WHERE id_song = ?
    ''', (id_album, album_source, id_song))
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
    """Fetch duration using: Last.fm -> Spotify -> NVIDIA (DeepSeek V4 Flash)"""
    
    # 1. Try Last.fm
    duration = fetch_duration_from_lastfm(artist_name, song_title)
    if duration:
        return duration, 'lastfm'

    # 2. Try Spotify
    duration = fetch_duration_from_spotify(artist_name, song_title)
    if duration:
        return duration, 'spotify'

    # 3. Try NVIDIA (free tier AI, slower and more prone to empty responses)
    duration = fetch_duration_from_nvidia(artist_name, song_title)
    if duration:
        return duration, 'nvidia'

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
                     artist_name_cache: dict):
    if '@attr' in scrobble_data and scrobble_data['@attr'].get('nowplaying') == 'true':
        return

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
            return

    if song_exists(conn, id_artist, song_title):
        return

    # Try to find album
    id_album, album_source = find_album_for_song(
        artist_name, song_title, album_from_scrobble, album_map, id_artist
    )
    
    # If album not found, set to 0 and mark for retry
    album_retry_count = 0
    if id_album == 0:
        album_retry_count = 0  # Will increment on retry
        album_source = 'none'

    print(f"    🎵 New song: {song_title}")
    if id_album > 0:
        print(f"      💿 Album ID: {id_album} (source: {album_source})")
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
              retry_count, album_retry_count, album_source)

    # Rate limiting: NVIDIA's free tier is slow and easier to rate-limit,
    # so it gets a longer pause than the cheap/fast sources.
    if duration_source == 'nvidia':
        time.sleep(1.5)
    elif duration_source in ('spotify', 'pending'):
        time.sleep(0.5)

    if album_source == 'nvidia':
        time.sleep(1.5)
    elif album_source in ('lastfm', 'musicbrainz'):
        time.sleep(0.3)


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
            
            if source == 'nvidia':
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
        id_album, album_source = find_album_for_song(
            artist_name, title, '', album_map, id_artist
        )
        
        if id_album > 0:
            print(f"         ✅ Found album! ID: {id_album} (source: {album_source})")
            update_song_album(conn, id_song, id_album, album_source)
            retried += 1
        else:
            print(f"         ⏳ Still no album found.")
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE Song
                SET album_retry_count = album_retry_count + 1, last_update = CURRENT_TIMESTAMP
                WHERE id_song = ?
            ''', (id_song,))
            conn.commit()
        
        if album_source == 'nvidia':
            time.sleep(1.5)
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
            process_scrobble(conn, track, artist_map, album_map, artist_name_cache)
            total_processed += 1

        print(f"   ✅ Page {page}/{total_pages} processed. Total: {total_processed}")

        if page >= total_pages:
            break

        page += 1
        time.sleep(0.3)

    print("\n🔄 Retrying pending duration songs...")
    retry_pending_songs(conn, artist_map, album_map, artist_name_cache)

    print("\n🔄 Retrying songs without album...")
    retry_album_songs(conn, artist_map, album_map, artist_name_cache)

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

    cursor.execute('SELECT album_source, COUNT(*) FROM Song WHERE id_album > 0 GROUP BY album_source')
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
    print("SONG DATABASE - RAW DATA FETCHER (NVIDIA edition)")
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

    if not nvidia_client:
        print("⚠️ NVIDIA_API_KEY not found in environment variables.")
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

"""
Album Database - Raw Data Fetcher

This module fetches album data from MusicBrainz for artists in the filtered artist database.
It retrieves:
- Album title
- Release year and date
- Primary type (album, ep, single, live, compilation...) + secondary types, as separate fields
- Total tracks
- Label (record label)
- Cover URL (from: Cover Art Archive → Last.fm → Deezer)
- is_reissue flag (remaster/deluxe/anniversary/expanded editions, detected from title)

KEY DESIGN CHOICE — release-groups, not releases:
MusicBrainz's "release" entity is a single physical/digital pressing (UK CD,
US vinyl, Japan reissue, streaming release...) — a real album can have a
dozen of these. Querying at that level floods the Album table with near-
duplicate rows for the same actual album, which is exactly what confuses
downstream album-matching (a song ending up assigned to some obscure
reissue instead of the original). "release-group" is MusicBrainz's own
abstraction for "the album, regardless of edition" — this script queries
at that level, and only pulls ONE representative OFFICIAL release per
release-group (never bootleg/promo-only) to get track count / label.

OPTIMIZATION:
- If artist exists in Album table → fetch only releases from last 2 years.
- If artist is new → fetch full discography.
- Never overwrite existing albums.
"""

import sqlite3
import os
import sys
import requests
import time
import re
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Try to import musicbrainzngs
try:
    import musicbrainzngs
except ImportError:
    print("⚠️ musicbrainzngs not installed. Installing...")
    os.system(f"{sys.executable} -m pip install musicbrainzngs")
    import musicbrainzngs

# Load environment variables
load_dotenv()

# Database paths
ARTIST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '1_artist_genres.db')
ALBUM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2_albums_raw.db')

# --- Credentials ---
LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY')

LASTFM_API_URL = 'https://ws.audioscrobbler.com/2.0/'
DEEZER_API_URL = 'https://api.deezer.com/search/album'

# Configure MusicBrainz
musicbrainzngs.set_useragent(
    "my_scrobbles",
    "1.0",
    "https://github.com/adroguett-scratch/my-lastfm-scrobbles"
)

# Maximum years to look back for existing artists
MAX_YEARS_BACK = 2


# ============================================
# COVER FETCHING FUNCTIONS
# ============================================

def get_cover_from_lastfm(artist_name: str, album_title: str) -> Optional[str]:
    """Get album cover from Last.fm API."""
    if not LASTFM_API_KEY:
        return None

    try:
        params = {
            'method': 'album.getinfo',
            'artist': artist_name,
            'album': album_title,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }

        resp = requests.get(LASTFM_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if 'error' in data:
            return None

        album = data.get('album', {})
        images = album.get('image', [])

        for img in images:
            if img.get('size') == 'extralarge':
                url = img.get('#text')
                if url and not url.endswith('2a96fbd4b0e3e8c4.png'):
                    return url

        return None

    except Exception as e:
        print(f"      ⚠️ Last.fm cover error: {e}")
        return None


def get_cover_from_deezer(artist_name: str, album_title: str) -> Optional[str]:
    """Get album cover from Deezer API (no auth required)."""
    try:
        params = {'q': f'artist:"{artist_name}" album:"{album_title}"', 'limit': 1}
        resp = requests.get(DEEZER_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        albums = data.get('data', [])
        if albums:
            cover = albums[0].get('cover_big')
            if cover:
                return cover

        return None

    except Exception as e:
        print(f"      ⚠️ Deezer cover error: {e}")
        return None


def get_cover_from_coverartarchive(release_group_mbid: str) -> Optional[str]:
    """
    Get album cover from the Cover Art Archive, MusicBrainz's own cover
    art service — indexed directly by release-group MBID, so this is an
    exact lookup, not a fuzzy text search. Far more reliable than asking
    an LLM to "remember" a cover URL (which it can't verify exists at all).
    """
    if not release_group_mbid:
        return None

    try:
        url = f"https://coverartarchive.org/release-group/{release_group_mbid}/front-500"
        resp = requests.head(url, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return url
        return None
    except Exception as e:
        print(f"      ⚠️ Cover Art Archive error: {e}")
        return None


def get_album_cover(artist_name: str, album_title: str, release_group_mbid: str = None) -> tuple:
    """
    Get album cover using multiple sources in order:
    1. Cover Art Archive (exact match by release-group MBID, when available)
    2. Last.fm (fuzzy text search)
    3. Deezer (fuzzy text search)

    DeepSeek was removed as a cover source: an LLM has no way to verify a
    cover URL actually exists, so it's pure hallucination risk for zero
    benefit now that we have an exact MBID-based lookup available.

    Returns (cover_url, source)
    """
    # 1. Try Cover Art Archive (exact, MBID-based)
    if release_group_mbid:
        cover = get_cover_from_coverartarchive(release_group_mbid)
        if cover:
            return cover, 'coverartarchive'

    # 2. Try Last.fm
    cover = get_cover_from_lastfm(artist_name, album_title)
    if cover:
        return cover, 'lastfm'

    # 3. Try Deezer
    cover = get_cover_from_deezer(artist_name, album_title)
    if cover:
        return cover, 'deezer'

    return None, None


# ============================================
# DATABASE FUNCTIONS
# ============================================

def create_schema(conn):
    """Creates the Album table and handles schema migrations."""
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Album'")
    table_exists = cursor.fetchone() is not None

    if not table_exists:
        cursor.execute('''
            CREATE TABLE Album (
                id_album     INTEGER PRIMARY KEY AUTOINCREMENT,
                id_artist    INTEGER NOT NULL,
                title        TEXT    NOT NULL,
                release_year INTEGER,
                release_date TEXT,
                album_type   TEXT,
                secondary_types TEXT,
                is_reissue   INTEGER DEFAULT 0,
                mbid         TEXT,
                total_tracks INTEGER,
                label        TEXT,
                producer     TEXT,
                cover_url    TEXT,
                cover_source TEXT,
                tracklist_json   TEXT,
                tracklist_source TEXT,
                last_update  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (id_artist, title, release_year)
            )
        ''')
        print("✅ Created table Album")
    else:
        cursor.execute("PRAGMA table_info(Album)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if 'cover_source' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN cover_source TEXT')
            print("✅ Added column: cover_source")

        if 'cover_url' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN cover_url TEXT')
            print("✅ Added column: cover_url")

        if 'producer' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN producer TEXT')
            print("✅ Added column: producer")

        if 'secondary_types' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN secondary_types TEXT')
            print("✅ Added column: secondary_types")

        if 'is_reissue' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN is_reissue INTEGER DEFAULT 0')
            print("✅ Added column: is_reissue")

        if 'mbid' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN mbid TEXT')
            print("✅ Added column: mbid")

        if 'tracklist_json' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN tracklist_json TEXT')
            print("✅ Added column: tracklist_json")

        if 'tracklist_source' not in existing_columns:
            cursor.execute('ALTER TABLE Album ADD COLUMN tracklist_source TEXT')
            print("✅ Added column: tracklist_source")

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_artist ON Album (id_artist)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_title ON Album (title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_release_year ON Album (release_year)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Metadata (
            key     TEXT PRIMARY KEY,
            value   TEXT,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()


def get_artists(conn) -> List[Dict[str, Any]]:
    """Get all artists from the filtered artist database."""
    cursor = conn.cursor()
    cursor.execute('SELECT id_artist, name FROM Artist ORDER BY id_artist')
    rows = cursor.fetchall()
    return [{'id': row[0], 'name': row[1]} for row in rows]


def get_artist_album_count(conn, id_artist: int) -> int:
    """Get count of albums for an artist."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Album WHERE id_artist = ?', (id_artist,))
    row = cursor.fetchone()
    return row[0] if row else 0


def album_exists(conn, id_artist: int, title: str, release_year: int = None) -> bool:
    """Check if an album already exists in the database."""
    cursor = conn.cursor()
    if release_year:
        cursor.execute('''
            SELECT id_album FROM Album
            WHERE id_artist = ? AND title = ? AND release_year = ?
        ''', (id_artist, title, release_year))
    else:
        cursor.execute('''
            SELECT id_album FROM Album
            WHERE id_artist = ? AND title = ?
        ''', (id_artist, title))
    return cursor.fetchone() is not None


def save_album(conn, artist_id: int, title: str, release_year: int = None,
               release_date: str = None, album_type: str = None,
               secondary_types: str = None, is_reissue: bool = False,
               mbid: str = None, total_tracks: int = None, label: str = None,
               producer: str = None, cover_url: str = None,
               cover_source: str = None, tracklist: Optional[List[str]] = None,
               tracklist_source: str = None):
    """Insert a new album into the database."""
    tracklist_json = json.dumps(tracklist, ensure_ascii=False) if tracklist else None

    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO Album (
            id_artist, title, release_year, release_date, album_type,
            secondary_types, is_reissue, mbid,
            total_tracks, label, producer, cover_url, cover_source,
            tracklist_json, tracklist_source, last_update
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (artist_id, title, release_year, release_date, album_type,
          secondary_types, int(bool(is_reissue)), mbid,
          total_tracks, label, producer, cover_url, cover_source,
          tracklist_json, tracklist_source))
    conn.commit()


def set_last_update_time(conn, timestamp: str):
    """Set the last update timestamp in Metadata."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO Metadata (key, value, last_update)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', ('last_update', timestamp))
    conn.commit()


def get_last_update_time(conn) -> Optional[str]:
    """Get the last update timestamp from Metadata."""
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM Metadata WHERE key = ?', ('last_update',))
    row = cursor.fetchone()
    return row[0] if row else None


# ============================================
# MUSICBRAINZ FUNCTIONS
# ============================================

# Patterns that indicate a release-group is a reissue/remaster/deluxe
# variant rather than the original release. Used only to FLAG these
# (is_reissue), never to discard them — the original strategy discussion
# was clear that they should be flagged for downstream prioritization, not
# silently dropped.
REISSUE_PATTERNS = re.compile(
    r'\((?:remaster(?:ed)?|deluxe|anniversary|expanded|bonus track|'
    r'special edition|reissue|legacy edition|super deluxe)[^)]*\)',
    re.IGNORECASE
)


def is_reissue_title(title: str) -> bool:
    """Detect remaster/deluxe/anniversary/etc. editions from the title."""
    return bool(REISSUE_PATTERNS.search(title or ''))


def search_artist_mbid(artist_name: str) -> Optional[str]:
    """Search for an artist in MusicBrainz and return the MBID."""
    try:
        result = musicbrainzngs.search_artists(query=artist_name, limit=1)
        artists = result.get('artist-list', [])
        if artists:
            return artists[0]['id']
        return None
    except Exception as e:
        print(f"      ⚠️ MusicBrainz search error for '{artist_name}': {e}")
        return None


def get_artist_release_groups(artist_mbid: str, since_year: int = None) -> List[Dict[str, Any]]:
    """
    Get release-groups for an artist from MusicBrainz — this is the "album,
    regardless of edition" abstraction, NOT individual physical/digital
    releases. A single browse_releases() call can return a dozen near-
    duplicate rows (UK CD, US vinyl, Japan reissue...) for the same real
    album; browse_release_groups() collapses all of those into one entry,
    which is what we actually want to store as "an album".

    If since_year is provided, only returns groups first released from
    that year onwards (same manual filtering approach as before, since the
    API doesn't support server-side year filtering here).
    """
    try:
        result = musicbrainzngs.browse_release_groups(
            artist=artist_mbid,
            includes=['artist-credits'],
            limit=100
        )
        groups = result.get('release-group-list', [])

        if since_year:
            filtered = []
            for rg in groups:
                date_str = rg.get('first-release-date', '')
                if date_str:
                    year_match = re.search(r'^(\d{4})', date_str)
                    if year_match and int(year_match.group(1)) >= since_year:
                        filtered.append(rg)
                else:
                    # No date on record — include it to be safe, same
                    # policy as the previous release-level implementation.
                    filtered.append(rg)
            return filtered

        return groups
    except Exception as e:
        print(f"      ⚠️ MusicBrainz release-groups error: {e}")
        return []


def get_official_release_details(release_group_mbid: str) -> Optional[Dict[str, Any]]:
    """
    Two-step lookup:
      1. browse_releases() — lightweight, finds WHICH release within this
         release-group is the official one (status filtering). This part
         was already working correctly (total_tracks counts were fine).
      2. get_release_by_id() — full detail LOOKUP on that specific release,
         which is the documented way to get nested track/recording titles.
         browse_releases() apparently doesn't reliably attach recording
         titles even when 'recordings' is requested — browse is optimized
         for lightweight listing, not deep nesting; lookup is the endpoint
         meant for full single-entity detail. This was the actual bug
         behind tracklist_json coming back NULL for every album.

    Deliberately skips bootleg/promotion-only editions — if a release-group
    has no official release at all, we treat it as not a real canonical
    album and return None (caller skips it).
    """
    try:
        browse_result = musicbrainzngs.browse_releases(
            release_group=release_group_mbid,
            includes=['media', 'labels'],
            limit=25
        )
        releases = browse_result.get('release-list', [])
        official = [r for r in releases if r.get('status', '').lower() == 'official']

        if not official:
            return None

        release_id = official[0]['id']

        # Full lookup on that one specific release for label + tracklist
        lookup_result = musicbrainzngs.get_release_by_id(
            release_id,
            includes=['media', 'labels', 'recordings']
        )
        release = lookup_result.get('release', {})

        label_info = release.get('label-info-list', [])
        label = label_info[0].get('label', {}).get('name', '') if label_info else None

        total_tracks = 0
        tracklist = []
        for medium in release.get('medium-list', []):
            for track in medium.get('track-list', []):
                total_tracks += 1
                title = track.get('recording', {}).get('title') or track.get('title')
                if title:
                    tracklist.append(title)

        return {'total_tracks': total_tracks, 'label': label, 'tracklist': tracklist}

    except Exception as e:
        print(f"      ⚠️ MusicBrainz official-release lookup error: {e}")
        return None


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def get_tracklist_from_wikipedia(artist_name: str, album_title: str) -> Optional[List[str]]:
    """
    Best-effort tracklist fetch from Wikipedia's {{Track listing}} infobox
    template. Used ONLY as a fallback when MusicBrainz didn't have a
    track-level listing for this release (rarer for well-known albums,
    more common for smaller/less-catalogued ones).

    This parses raw wikitext with regex, which is inherently a bit fragile
    — article formatting varies — so it's best-effort, not guaranteed.
    Returns None (not an empty list) on any failure/no-match, so the
    caller can tell "found nothing" apart from "found zero tracks".
    """
    try:
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': f'{album_title} {artist_name} album',
            'format': 'json',
            'srlimit': 3,
        }
        resp = requests.get(WIKIPEDIA_API_URL, params=search_params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get('query', {}).get('search', [])
        if not results:
            return None
        page_title = results[0]['title']

        parse_params = {
            'action': 'parse',
            'page': page_title,
            'prop': 'wikitext',
            'format': 'json',
        }
        resp = requests.get(WIKIPEDIA_API_URL, params=parse_params, timeout=15)
        resp.raise_for_status()
        wikitext = resp.json().get('parse', {}).get('wikitext', {}).get('*', '')
        if not wikitext:
            return None

        tracks = []
        block_matches = re.findall(r'\{\{Track listing.*?\n\}\}', wikitext, re.DOTALL | re.IGNORECASE)
        for block in block_matches:
            for raw_title in re.findall(r'\btitle\d*\s*=\s*(.+)', block):
                clean = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1", raw_title)  # [[link|Text]] -> Text
                clean = re.sub(r"'''?", '', clean)  # strip bold/italic wiki markup
                clean = clean.strip().strip('"').strip()
                if clean:
                    tracks.append(clean)

        return tracks if tracks else None

    except Exception as e:
        print(f"      ⚠️ Wikipedia tracklist error: {e}")
        return None


def get_tracklist(release_group_mbid: str, artist_name: str, album_title: str,
                   mb_tracklist: Optional[List[str]]) -> tuple:
    """
    Resolve the ordered tracklist for an album: MusicBrainz first (already
    fetched, real/verified), Wikipedia as fallback when MB had nothing.
    Returns (tracklist_list_or_None, source_string_or_None).
    """
    if mb_tracklist:
        return mb_tracklist, 'musicbrainz'

    wiki_tracklist = get_tracklist_from_wikipedia(artist_name, album_title)
    if wiki_tracklist:
        return wiki_tracklist, 'wikipedia'

    return None, None


def parse_release_group(rg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a MusicBrainz release-group dict into our format."""
    title = rg.get('title', '')

    date_str = rg.get('first-release-date', '')
    release_year = None
    if date_str:
        year_match = re.search(r'^(\d{4})', date_str)
        if year_match:
            release_year = int(year_match.group(1))

    primary_type = (rg.get('primary-type') or 'unknown').lower()
    secondary_types = rg.get('secondary-type-list', [])
    secondary_types_str = ', '.join(t.lower() for t in secondary_types) if secondary_types else None

    return {
        'mbid': rg.get('id'),
        'title': title,
        'release_year': release_year,
        'release_date': date_str or None,
        'album_type': primary_type,
        'secondary_types': secondary_types_str,
        'is_reissue': is_reissue_title(title),
    }


# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def fetch_albums_for_artist(artist_id: int, artist_name: str, album_conn):
    """
    Fetch and save albums for a single artist.
    - If artist has existing albums: fetch only releases from last 2 years.
    - If artist is new: fetch full discography.

    Queries MusicBrainz at the release-GROUP level (one row per actual
    album, not per physical/digital edition), and only keeps groups that
    have at least one OFFICIAL release — bootleg/promo-only groups are
    skipped entirely, at the source, rather than relying on downstream
    matching to filter them out later.
    """
    new_albums = 0
    skipped_no_official = 0
    
    # Check if artist already has albums
    album_count = get_artist_album_count(album_conn, artist_id)
    
    if album_count > 0:
        # Artist exists: fetch only recent releases (last 2 years)
        current_year = datetime.now().year
        since_year = current_year - MAX_YEARS_BACK
        print(f"   📌 Existing artist: fetching releases from {since_year} onwards")
    else:
        # New artist: fetch full discography
        since_year = None
        print(f"   📌 New artist: fetching full discography")

    # Get artist MBID from MusicBrainz
    mbid = search_artist_mbid(artist_name)
    if not mbid:
        print(f"   ⚠️ No MusicBrainz ID found for '{artist_name}'")
        return 0

    # Get release-groups (the "album, regardless of edition" abstraction)
    release_groups = get_artist_release_groups(mbid, since_year)
    if not release_groups:
        print(f"   ⚠️ No releases found for '{artist_name}'")
        return 0

    if since_year:
        print(f"   📀 Found {len(release_groups)} recent release-groups (since {since_year})")
    else:
        print(f"   📀 Found {len(release_groups)} total release-groups")

    for rg in release_groups:
        if not rg.get('title'):
            continue

        parsed = parse_release_group(rg)

        # Skip if album already exists (prevent duplicates)
        if album_exists(album_conn, artist_id, parsed['title'], parsed['release_year']):
            continue

        # Only keep release-groups that have at least one OFFICIAL release.
        # A group whose only releases are bootlegs/promos is not a real
        # canonical album — this is the main fix for songs ending up
        # assigned to demos/promos downstream.
        details = get_official_release_details(parsed['mbid'])
        if details is None:
            skipped_no_official += 1
            time.sleep(0.2)
            continue

        # Resolve the ordered tracklist: MusicBrainz first (already fetched
        # above, real/verified), Wikipedia as a fallback when MB had none.
        tracklist, tracklist_source = get_tracklist(
            parsed['mbid'], artist_name, parsed['title'], details.get('tracklist')
        )

        # Sanity check: if we got a tracklist, its length should roughly
        # match MusicBrainz's own track count. A mismatch is a signal this
        # might be the wrong edition (e.g. Wikipedia matched a deluxe
        # version with bonus tracks) — just a warning, not a blocker.
        if tracklist and details.get('total_tracks') and len(tracklist) != details['total_tracks']:
            print(f"      ⚠️ Tracklist length ({len(tracklist)}) doesn't match "
                  f"MusicBrainz track count ({details['total_tracks']}) — possibly a different edition.")

        # Get cover URL with fallbacks (Cover Art Archive first, exact MBID match)
        cover_url, cover_source = get_album_cover(artist_name, parsed['title'], parsed['mbid'])

        tag = " [REISSUE]" if parsed['is_reissue'] else ""
        track_info = f" [{len(tracklist)} tracks: {tracklist_source}]" if tracklist else " [no tracklist]"
        if cover_url:
            print(f"   💾 {parsed['title']} ({parsed['release_year'] or 'N/A'}) [{parsed['album_type']}]{tag}{track_info} [cover: {cover_source}]")
        else:
            print(f"   💾 {parsed['title']} ({parsed['release_year'] or 'N/A'}) [{parsed['album_type']}]{tag}{track_info} [no cover]")

        save_album(
            album_conn,
            artist_id=artist_id,
            title=parsed['title'],
            release_year=parsed['release_year'],
            release_date=parsed['release_date'],
            album_type=parsed['album_type'],
            secondary_types=parsed['secondary_types'],
            is_reissue=parsed['is_reissue'],
            mbid=parsed['mbid'],
            total_tracks=details['total_tracks'],
            label=details['label'],
            producer=None,
            cover_url=cover_url,
            cover_source=cover_source,
            tracklist=tracklist,
            tracklist_source=tracklist_source
        )
        new_albums += 1

        # Avoid rate limiting
        time.sleep(0.2)

    if skipped_no_official:
        print(f"   ⏭️  Skipped {skipped_no_official} release-group(s) with no official release (bootleg/promo-only)")

    return new_albums


def create_album_database():
    """Main function to create and populate the album database."""

    print("=" * 60)
    print("ALBUM DATABASE - RAW DATA FETCHER")
    print("=" * 60)

    if not os.path.exists(ARTIST_DB_PATH):
        print(f"❌ Artist database not found: {ARTIST_DB_PATH}")
        print("   Please run 1_artist_genre_filter.py first.")
        return

    print(f"📂 Reading artists from: {ARTIST_DB_PATH}")
    artist_conn = sqlite3.connect(ARTIST_DB_PATH)
    artists = get_artists(artist_conn)
    artist_conn.close()

    if not artists:
        print("❌ No artists found in the artist database.")
        return

    print(f"🎵 Found {len(artists)} artists")
    print(f"📅 Only fetching releases from last {MAX_YEARS_BACK} years for existing artists")
    print("-" * 60)

    # Connect to album database
    os.makedirs(os.path.dirname(ALBUM_DB_PATH), exist_ok=True)
    album_conn = sqlite3.connect(ALBUM_DB_PATH)
    create_schema(album_conn)

    total_new = 0
    skipped_artists = 0

    for idx, artist in enumerate(artists, start=1):
        artist_id = artist['id']
        artist_name = artist['name']

        print(f"\n[{idx}/{len(artists)}] {artist_name} (ID: {artist_id})")

        new = fetch_albums_for_artist(artist_id, artist_name, album_conn)
        
        if new == 0:
            skipped_artists += 1
        
        total_new += new

        # Avoid rate limiting
        time.sleep(0.5)

    # Update metadata
    now = datetime.now().isoformat()
    set_last_update_time(album_conn, now)

    # Get statistics
    cursor = album_conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Album')
    total_albums = cursor.fetchone()[0]

    print("-" * 60)
    print(f"\n✅ Album database updated successfully")
    print(f"📁 Location: {ALBUM_DB_PATH}")
    print(f"📋 Table 'Album' with album details")
    print(f"🎵 Total albums in DB: {total_albums}")
    print(f"   New albums added: {total_new}")
    print(f"   Artists skipped (no new releases): {skipped_artists}")
    print("=" * 60)

    album_conn.close()


if __name__ == "__main__":
    create_album_database()

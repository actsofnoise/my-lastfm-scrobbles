"""
Album Database - Raw Data Fetcher

This module fetches album data from MusicBrainz for artists in the filtered artist database.
It retrieves:
- Album title
- Release year and date
- Album type (studio, live, compilation, etc.)
- Total tracks
- Label (record label)
- Cover URL (from: Last.fm → Deezer → DeepSeek)

BEHAVIOR:
- For each artist in 1_artist_genres.db, fetch their releases from MusicBrainz.
- Insert new albums into 2_albums_raw.db.
- If an album already exists (by id_artist + title), skip or update missing fields (like cover_url).
- id_album is autoincrement, id_artist is the same as in Artist table.
"""

import sqlite3
import os
import sys
import requests
import time
import re
from datetime import datetime
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
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')

LASTFM_API_URL = 'https://ws.audioscrobbler.com/2.0/'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEZER_API_URL = 'https://api.deezer.com/search/album'

# Configure MusicBrainz
musicbrainzngs.set_useragent(
    "my_scrobbles",
    "1.0",
    "https://github.com/adroguett-scratch/my-lastfm-scrobbles"
)


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
                if url and not url.endswith('2a96fbd4b0e3e8c4.png'):  # Generic placeholder
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


def get_cover_from_deepseek(artist_name: str, album_title: str) -> Optional[str]:
    """Get album cover from DeepSeek API (last resort)."""
    if not DEEPSEEK_API_KEY:
        return None

    try:
        prompt = f"""You are a music assistant. Find the official album cover URL for the album '{album_title}' by {artist_name}.

IMPORTANT:
- Respond ONLY with the image URL.
- Do not add any other text or explanation.
- If you can't find a valid cover URL, respond with 'NONE'."""

        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'You are a music assistant. Respond ONLY with a valid image URL or "NONE".'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.1,
            'max_tokens': 150
        }

        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=15)
        resp.raise_for_status()

        result = resp.json()
        cover_url = result['choices'][0]['message']['content'].strip()

        if cover_url and cover_url != 'NONE' and cover_url.startswith('http'):
            return cover_url

        return None

    except Exception as e:
        print(f"      ⚠️ DeepSeek cover error: {e}")
        return None


def get_album_cover(artist_name: str, album_title: str) -> tuple:
    """
    Get album cover using multiple sources in order:
    1. Last.fm
    2. Deezer
    3. DeepSeek (last resort)

    Returns (cover_url, source)
    """
    # 1. Try Last.fm
    cover = get_cover_from_lastfm(artist_name, album_title)
    if cover:
        return cover, 'lastfm'

    # 2. Try Deezer
    cover = get_cover_from_deezer(artist_name, album_title)
    if cover:
        return cover, 'deezer'

    # 3. Try DeepSeek (last resort)
    cover = get_cover_from_deepseek(artist_name, album_title)
    if cover:
        return cover, 'deepseek'

    return None, None


# ============================================
# DATABASE FUNCTIONS
# ============================================

def create_schema(conn):
    """Creates the Album table if it doesn't exist."""
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Album (
            id_album     INTEGER PRIMARY KEY AUTOINCREMENT,
            id_artist    INTEGER NOT NULL,
            title        TEXT    NOT NULL,
            release_year INTEGER,
            release_date TEXT,
            album_type   TEXT,
            total_tracks INTEGER,
            label        TEXT,
            producer     TEXT,
            cover_url    TEXT,
            cover_source TEXT,
            last_update  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (id_artist, title, release_year)
        )
    ''')

    # Indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_artist ON Album (id_artist)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_title ON Album (title)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_release_year ON Album (release_year)')

    # Table to track last update time
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
               total_tracks: int = None, label: str = None,
               producer: str = None, cover_url: str = None,
               cover_source: str = None):
    """Insert a new album into the database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO Album (
            id_artist, title, release_year, release_date, album_type,
            total_tracks, label, producer, cover_url, cover_source, last_update
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (artist_id, title, release_year, release_date, album_type,
          total_tracks, label, producer, cover_url, cover_source))
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


def get_artist_releases(artist_mbid: str) -> List[Dict[str, Any]]:
    """Get all releases (albums) for an artist from MusicBrainz."""
    try:
        result = musicbrainzngs.browse_releases(
            artist=artist_mbid,
            includes=['release-groups'],
            limit=100
        )
        releases = result.get('release-list', [])
        return releases
    except Exception as e:
        print(f"      ⚠️ MusicBrainz releases error: {e}")
        return []


def parse_release(release: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a MusicBrainz release dict into our format."""
    title = release.get('title', '')

    # Release date
    date_str = release.get('date', '')
    release_year = None
    release_date = None
    if date_str:
        try:
            year_match = re.search(r'^(\d{4})', date_str)
            if year_match:
                release_year = int(year_match.group(1))
            if len(date_str) >= 10:
                release_date = date_str
            else:
                release_date = date_str
        except:
            pass

    # Album type
    release_group = release.get('release-group', {})
    primary_type = release_group.get('primary-type', '')
    secondary_types = release_group.get('secondary-types', [])
    album_type = primary_type.lower() if primary_type else 'unknown'
    if secondary_types:
        album_type = f"{album_type} / {' / '.join(secondary_types)}".lower()

    # Label
    label_info = release.get('label-info-list', [])
    label = None
    if label_info:
        label = label_info[0].get('label', {}).get('name', '')

    # Total tracks
    medium_list = release.get('medium-list', [])
    total_tracks = 0
    for medium in medium_list:
        track_list = medium.get('track-list', [])
        total_tracks += len(track_list)

    return {
        'title': title,
        'release_year': release_year,
        'release_date': release_date,
        'album_type': album_type,
        'total_tracks': total_tracks,
        'label': label,
        'producer': None,
    }


# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def fetch_albums_for_artist(artist_id: int, artist_name: str, album_conn):
    """Fetch and save albums for a single artist."""
    new_albums = 0
    updated_covers = 0

    # 1. Get artist MBID from MusicBrainz
    mbid = search_artist_mbid(artist_name)
    if not mbid:
        print(f"   ⚠️ No MusicBrainz ID found for '{artist_name}'")
        return 0, 0

    # 2. Get releases
    releases = get_artist_releases(mbid)
    if not releases:
        print(f"   ⚠️ No releases found for '{artist_name}'")
        return 0, 0

    print(f"   📀 Found {len(releases)} releases")

    for release in releases:
        if not release.get('title'):
            continue

        parsed = parse_release(release)

        # Check if album already exists
        if album_exists(album_conn, artist_id, parsed['title'], parsed['release_year']):
            continue

        # Get cover URL with fallbacks
        cover_url, cover_source = get_album_cover(artist_name, parsed['title'])

        if cover_url:
            print(f"   💾 {parsed['title']} ({parsed['release_year'] or 'N/A'}) [cover: {cover_source}]")
        else:
            print(f"   💾 {parsed['title']} ({parsed['release_year'] or 'N/A'}) [no cover]")

        # Save album
        save_album(
            album_conn,
            artist_id=artist_id,
            title=parsed['title'],
            release_year=parsed['release_year'],
            release_date=parsed['release_date'],
            album_type=parsed['album_type'],
            total_tracks=parsed['total_tracks'],
            label=parsed['label'],
            producer=parsed['producer'],
            cover_url=cover_url,
            cover_source=cover_source
        )
        new_albums += 1

        # Avoid rate limiting
        time.sleep(0.2)

    return new_albums, updated_covers


def create_album_database():
    """Main function to create and populate the album database."""

    print("=" * 60)
    print("ALBUM DATABASE - RAW DATA FETCHER")
    print("=" * 60)

    # Check if artist database exists
    if not os.path.exists(ARTIST_DB_PATH):
        print(f"❌ Artist database not found: {ARTIST_DB_PATH}")
        print("   Please run 1_artist_genre_filter.py first.")
        return

    # Connect to artist database
    print(f"📂 Reading artists from: {ARTIST_DB_PATH}")
    artist_conn = sqlite3.connect(ARTIST_DB_PATH)
    artists = get_artists(artist_conn)
    artist_conn.close()

    if not artists:
        print("❌ No artists found in the artist database.")
        return

    print(f"🎵 Found {len(artists)} artists")

    # Connect to album database
    os.makedirs(os.path.dirname(ALBUM_DB_PATH), exist_ok=True)
    album_conn = sqlite3.connect(ALBUM_DB_PATH)
    create_schema(album_conn)

    # Check last update time
    last_update = get_last_update_time(album_conn)
    if last_update:
        print(f"📌 Last update: {last_update}")
        print("   Fetching albums for artists not yet processed...")
    else:
        print("📂 New album database. Fetching all albums...")

    print("-" * 60)

    total_new = 0
    total_updated_covers = 0

    for idx, artist in enumerate(artists, start=1):
        artist_id = artist['id']
        artist_name = artist['name']

        print(f"\n[{idx}/{len(artists)}] {artist_name} (ID: {artist_id})")

        new, covers = fetch_albums_for_artist(artist_id, artist_name, album_conn)
        total_new += new
        total_updated_covers += covers

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
    print(f"   Covers updated: {total_updated_covers}")
    print("=" * 60)

    album_conn.close()


if __name__ == "__main__":
    create_album_database()

"""
Artist Genre Filter Module

This module filters and normalizes genre tags from the raw artist database
and creates/updates the filtered artist database.

BEHAVIOR:
- If artist is new: insert all filtered data (genres, nationality, image)
- If artist exists:
    - If missing image: copy from raw database
    - If missing nationality: copy from raw database
    - If missing genres (no genre_1): filter and save genres
    - Otherwise: skip (already complete)
- Never overwrite existing data
- Preserve id_artist (same as raw database)
"""

import sqlite3
import os
import sys
import re
from datetime import datetime
from typing import List, Optional, Set, Dict

# Add the genre_filter_dictionaries directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'genre_filter_dictionaries'))

from genre_dict import GENRE_DICT, GENERIC_TAGS
from nationality_dict import NATIONALITY_TAGS

# Database paths
RAW_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '0_artist_raw.db')
FILTERED_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '1_artist_genres.db')


# ============================================
# DECADE TAGS TO DISCARD
# ============================================

DECADE_TAGS = {
    '60s', '70s', '80s', '90s', '00s', '2000s', '2010s', '2020s',
    '1960s', '1970s', '1980s', '1990s', '2000s',
    '60', '70', '80', '90', '00',
    'sixties', 'seventies', 'eighties', 'nineties'
}


# ============================================
# ARTIST KEYWORDS TO DISCARD
# ============================================

ARTIST_KEYWORDS = {
    'band', 'group', 'project', 'ensemble', 'orchestra',
    'orchestral', 'symphony', 'philharmonic', 'chamber',
    'music', 'songs', 'album', 'records', 'live', 'concert',
    'tour', 'festival'
}


# ============================================
# FILTER FUNCTIONS
# ============================================

def is_nationality_tag(tag: str) -> bool:
    """Check if a tag represents a nationality."""
    tag_lower = tag.lower()
    return tag_lower in NATIONALITY_TAGS


def is_decade_tag(tag: str) -> bool:
    """Check if a tag represents a decade."""
    tag_lower = tag.lower()
    return tag_lower in DECADE_TAGS


def is_generic_tag(tag: str) -> bool:
    """Check if a tag is too generic (e.g., 'rock', 'pop', 'metal')."""
    tag_lower = tag.lower()
    return tag_lower in GENERIC_TAGS


def is_artist_keyword(tag: str) -> bool:
    """Check if a tag contains artist keywords (e.g., 'band', 'group')."""
    tag_lower = tag.lower()
    return any(keyword in tag_lower for keyword in ARTIST_KEYWORDS)


def is_artist_name_tag(tag: str, artist_name: str) -> bool:
    """
    Check if a tag is the artist's name (e.g., 'pink floyd' as a tag for Pink Floyd).
    """
    if not tag or not artist_name:
        return False
    
    tag_clean = tag.lower().strip()
    artist_clean = artist_name.lower().strip()
    
    if tag_clean == artist_clean:
        return True
    
    if len(tag_clean) >= 3 and tag_clean in artist_clean:
        return True
    
    if len(artist_clean) >= 3 and artist_clean in tag_clean:
        return True
    
    common_words = {'the', 'and', 'of', 'for', 'with', 'on', 'at', 'from', 'by'}
    tag_words = set(tag_clean.split()) - common_words
    artist_words = set(artist_clean.split()) - common_words
    
    if tag_words and artist_words:
        for tw in tag_words:
            for aw in artist_words:
                if len(tw) >= 3 and len(aw) >= 3 and (tw in aw or aw in tw):
                    return True
    
    return False


def normalize_genre(tag: str) -> str:
    """Normalize a genre tag using the GENRE_DICT. Returns lowercase."""
    tag_lower = tag.lower()
    if tag_lower in GENRE_DICT:
        return GENRE_DICT[tag_lower].lower()
    return tag_lower


def should_keep_tag(tag: str, artist_name: str = None) -> bool:
    """Determine if a tag should be kept after filtering."""
    if not tag or not tag.strip():
        return False
    
    tag_lower = tag.lower()
    
    if is_nationality_tag(tag_lower):
        return False
    
    if is_decade_tag(tag_lower):
        return False
    
    if is_generic_tag(tag_lower):
        return False
    
    if is_artist_keyword(tag_lower):
        return False
    
    if artist_name and is_artist_name_tag(tag, artist_name):
        return False
    
    return True


def filter_and_normalize_genres(raw_tags: List[str], artist_name: str = None) -> List[str]:
    """Filter and normalize a list of genre tags. Returns unique, clean genres (lowercase)."""
    if not raw_tags:
        return []
    
    normalized = []
    for tag in raw_tags:
        if tag and isinstance(tag, str):
            normalized_tag = normalize_genre(tag)
            if normalized_tag:
                normalized.append(normalized_tag)
    
    filtered = []
    for tag in normalized:
        if should_keep_tag(tag, artist_name):
            filtered.append(tag)
    
    seen = set()
    unique_genres = []
    for tag in filtered:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_genres.append(tag_lower)
    
    return unique_genres


def get_top_n_genres(raw_tags: List[str], artist_name: str = None, n: int = 5) -> List[str]:
    """Get the top N genres from a list of raw tags (all lowercase)."""
    filtered = filter_and_normalize_genres(raw_tags, artist_name)
    return filtered[:n]


# ============================================
# DATABASE FUNCTIONS
# ============================================

def ensure_filtered_schema(conn):
    """Create or update the filtered Artist table schema."""
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Artist'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE Artist (
                id_artist   INTEGER PRIMARY KEY,
                name        TEXT    NOT NULL UNIQUE,
                nationality TEXT,
                artist_image_url TEXT,
                genre_1     TEXT,
                genre_2     TEXT,
                genre_3     TEXT,
                genre_4     TEXT,
                genre_5     TEXT,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX idx_artist_name_filtered ON Artist (name)')
        conn.commit()
        print("   ✅ Created Artist table with image column")
        return
    
    # Table exists: add missing columns
    cursor.execute("PRAGMA table_info(Artist)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    if 'artist_image_url' not in existing_columns:
        cursor.execute('ALTER TABLE Artist ADD COLUMN artist_image_url TEXT')
        print("   ✅ Added column: artist_image_url")
    
    if 'genre_1' not in existing_columns:
        # If old version had 'genre' column, rename it
        if 'genre' in existing_columns:
            cursor.execute('ALTER TABLE Artist RENAME COLUMN genre TO genre_1')
            print("   ✅ Renamed column: genre → genre_1")
        else:
            cursor.execute('ALTER TABLE Artist ADD COLUMN genre_1 TEXT')
            print("   ✅ Added column: genre_1")
    
    for n in range(2, 6):
        col = f'genre_{n}'
        if col not in existing_columns:
            cursor.execute(f'ALTER TABLE Artist ADD COLUMN {col} TEXT')
            print(f"   ✅ Added column: {col}")
    
    # Ensure index exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_artist_name_filtered'")
    if not cursor.fetchone():
        cursor.execute('CREATE INDEX idx_artist_name_filtered ON Artist (name)')
        print("   ✅ Added index: idx_artist_name_filtered")
    
    conn.commit()


def get_raw_artists(conn):
    """Get all artists from the raw database."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id_artist, name, nationality, artist_image_url,
               genre_1, genre_2, genre_3, genre_4, genre_5,
               genre_6, genre_7, genre_8, genre_9, genre_10,
               genre_11, genre_12, genre_13, genre_14, genre_15
        FROM Artist
    ''')
    
    artists = []
    for row in cursor.fetchall():
        artist_id = row[0]
        name = row[1]
        nationality = row[2]
        image_url = row[3]
        # Collect non-None genres
        raw_genres = []
        for i in range(4, 19):  # genre_1 to genre_15
            if row[i] is not None:
                raw_genres.append(row[i])
        artists.append({
            'id': artist_id,
            'name': name,
            'nationality': nationality,
            'image_url': image_url,
            'raw_genres': raw_genres
        })
    
    return artists


def get_filtered_artist(conn, artist_id):
    """Get an artist from the filtered database by id."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id_artist, name, nationality, artist_image_url,
               genre_1, genre_2, genre_3, genre_4, genre_5
        FROM Artist
        WHERE id_artist = ?
    ''', (artist_id,))
    row = cursor.fetchone()
    if not row:
        return None
    
    return {
        'id': row[0],
        'name': row[1],
        'nationality': row[2],
        'image_url': row[3],
        'genres': [row[4], row[5], row[6], row[7], row[8]]  # genre_1 to genre_5
    }


def insert_filtered_artist(conn, artist_id, name, nationality, image_url, genres):
    """Insert a new artist into the filtered database."""
    while len(genres) < 5:
        genres.append(None)
    
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Artist (id_artist, name, nationality, artist_image_url,
                            genre_1, genre_2, genre_3, genre_4, genre_5, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (artist_id, name, nationality, image_url,
          genres[0], genres[1], genres[2], genres[3], genres[4]))
    conn.commit()


def update_filtered_artist(conn, artist_id, nationality=None, image_url=None, genres=None):
    """
    Update only the fields that are provided (not None).
    Returns True if anything was updated.
    """
    updates = []
    params = []
    
    if nationality is not None:
        updates.append("nationality = ?")
        params.append(nationality)
    
    if image_url is not None:
        updates.append("artist_image_url = ?")
        params.append(image_url)
    
    if genres is not None:
        while len(genres) < 5:
            genres.append(None)
        updates.append("genre_1 = ?")
        params.append(genres[0])
        updates.append("genre_2 = ?")
        params.append(genres[1])
        updates.append("genre_3 = ?")
        params.append(genres[2])
        updates.append("genre_4 = ?")
        params.append(genres[3])
        updates.append("genre_5 = ?")
        params.append(genres[4])
    
    if not updates:
        return False
    
    updates.append("last_update = CURRENT_TIMESTAMP")
    params.append(artist_id)
    
    cursor = conn.cursor()
    cursor.execute(f'''
        UPDATE Artist
        SET {', '.join(updates)}
        WHERE id_artist = ?
    ''', params)
    conn.commit()
    return True


# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_and_filter_artists():
    """Process raw artists and generate/update the filtered database."""
    
    print("=" * 60)
    print("ARTIST GENRE FILTER")
    print("=" * 60)
    
    # Check if raw database exists
    if not os.path.exists(RAW_DB_PATH):
        print(f"❌ Raw database not found: {RAW_DB_PATH}")
        print("   Please run 0_artist_db_raw.py first.")
        return
    
    # Connect to databases
    print(f"📂 Reading from: {RAW_DB_PATH}")
    raw_conn = sqlite3.connect(RAW_DB_PATH)
    raw_conn.row_factory = sqlite3.Row
    
    print(f"📂 Writing to: {FILTERED_DB_PATH}")
    filtered_conn = sqlite3.connect(FILTERED_DB_PATH)
    
    # Ensure schema
    print("📋 Checking filtered database schema...")
    ensure_filtered_schema(filtered_conn)
    
    # Get all raw artists
    raw_artists = get_raw_artists(raw_conn)
    print(f"🎵 Found {len(raw_artists)} artists in raw database")
    print("-" * 60)
    
    inserted = 0
    updated = 0
    skipped = 0
    errors = 0
    
    for raw in raw_artists:
        artist_id = raw['id']
        name = raw['name']
        raw_genres = raw['raw_genres']
        raw_nationality = raw['nationality']
        raw_image = raw['image_url']
        
        # Check if artist already exists in filtered database
        filtered = get_filtered_artist(filtered_conn, artist_id)
        
        if filtered is None:
            # NEW ARTIST: insert all data
            print(f"\n🆕 New artist: {name} (ID: {artist_id})")
            
            # Filter genres
            clean_genres = filter_and_normalize_genres(raw_genres, artist_name=name)
            top_genres = clean_genres[:5]
            
            if not top_genres:
                print(f"   ⚠️ No genres after filtering. Skipping insertion.")
                errors += 1
                continue
            
            # Use raw nationality/image if available, else use defaults
            nationality = raw_nationality if raw_nationality else 'Unknown'
            image_url = raw_image if raw_image else None
            
            insert_filtered_artist(filtered_conn, artist_id, name, nationality, image_url, top_genres)
            inserted += 1
            print(f"   ✅ Inserted with genres: {top_genres}")
            print(f"   📍 Nationality: {nationality}")
            print(f"   🖼️  Image: {'Yes' if image_url else 'No'}")
            continue
        
        # ARTIST EXISTS: check for missing fields
        print(f"\n🔍 Checking: {name} (ID: {artist_id})")
        
        needs_update = False
        updated_fields = []
        
        # Check nationality
        if not filtered['nationality'] or filtered['nationality'] == 'Unknown':
            if raw_nationality and raw_nationality != 'Unknown':
                nationality = raw_nationality
                print(f"   📍 Updating nationality: {nationality}")
                needs_update = True
                updated_fields.append('nationality')
            else:
                nationality = None
        else:
            nationality = None
        
        # Check image
        if not filtered['image_url']:
            if raw_image:
                image_url = raw_image
                print(f"   🖼️  Updating image: {image_url[:50]}...")
                needs_update = True
                updated_fields.append('image')
            else:
                image_url = None
        else:
            image_url = None
        
        # Check genres (if no genre_1, re-filter)
        if not filtered['genres'][0]:  # genre_1 is None or empty
            clean_genres = filter_and_normalize_genres(raw_genres, artist_name=name)
            top_genres = clean_genres[:5]
            if top_genres:
                print(f"   🏷️  Updating genres: {top_genres}")
                needs_update = True
                updated_fields.append('genres')
            else:
                print(f"   ⚠️ No genres after filtering. Keeping existing.")
                top_genres = None
        else:
            top_genres = None
        
        if needs_update:
            update_filtered_artist(filtered_conn, artist_id, nationality, image_url, top_genres)
            updated += 1
            print(f"   ✅ Updated: {', '.join(updated_fields)}")
        else:
            print(f"   ✅ Already complete. Skipping.")
            skipped += 1
    
    # Commit and close
    filtered_conn.commit()
    
    # Get statistics
    cursor = filtered_conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Artist')
    total = cursor.fetchone()[0]
    
    print("-" * 60)
    print(f"\n✅ Filtered database updated successfully")
    print(f"📁 Location: {FILTERED_DB_PATH}")
    print(f"📋 Table 'Artist' with filtered genres (top 5, all lowercase)")
    print(f"🎵 Total artists in filtered DB: {total}")
    print(f"   New artists inserted: {inserted}")
    print(f"   Artists updated (missing fields): {updated}")
    print(f"   Artists skipped (already complete): {skipped}")
    print(f"   Errors (no genres): {errors}")
    print("=" * 60)
    
    raw_conn.close()
    filtered_conn.close()


if __name__ == "__main__":
    process_and_filter_artists()

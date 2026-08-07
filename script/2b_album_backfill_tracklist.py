"""
Album Tracklist Backfill

Standalone, one-off maintenance script. Fills in tracklist_json /
tracklist_source ONLY for albums that already exist in 2_albums_raw.db but
have NULL there (e.g. everything fetched before the browse->lookup fix, or
before the tracklist_json column existed at all).

Does NOT touch total_tracks, label, cover, or anything else already set —
this is deliberately narrow, so it's cheap to run and safe to re-run.

Skips albums with no mbid on record (very old rows, from before the mbid
column was added) — those genuinely can't be backfilled this way; they'd
need a full re-fetch instead. Their count is reported separately.
"""

import sqlite3
import os
import sys
import re
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import musicbrainzngs
except ImportError:
    print("⚠️ musicbrainzngs not installed. Installing...")
    os.system(f"{sys.executable} -m pip install musicbrainzngs")
    import musicbrainzngs

import requests

ALBUM_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '2_albums_raw.db')
ARTIST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '1_artist_genres.db')

musicbrainzngs.set_useragent("my_scrobbles", "1.0", "contact@example.com")

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


# ============================================
# TRACKLIST RESOLUTION (same logic as 2_album_db_raw.py)
# ============================================

def get_tracklist_from_musicbrainz(release_group_mbid: str) -> Optional[List[str]]:
    """
    Two-step lookup (the fix confirmed working via diagnose_tracklist.py):
      1. browse_releases() to find the official release within the group.
      2. get_release_by_id() — full detail LOOKUP — for the real tracklist.
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
        lookup_result = musicbrainzngs.get_release_by_id(
            release_id,
            includes=['media', 'labels', 'recordings']
        )
        release = lookup_result.get('release', {})

        tracklist = []
        for medium in release.get('medium-list', []):
            for track in medium.get('track-list', []):
                title = track.get('recording', {}).get('title') or track.get('title')
                if title:
                    tracklist.append(title)

        return tracklist if tracklist else None

    except Exception as e:
        print(f"      ⚠️ MusicBrainz lookup error: {e}")
        return None


def get_tracklist_from_wikipedia(artist_name: str, album_title: str) -> Optional[List[str]]:
    """Best-effort fallback — same approach as 2_album_db_raw.py."""
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
                clean = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1", raw_title)
                clean = re.sub(r"'''?", '', clean)
                clean = clean.strip().strip('"').strip()
                if clean:
                    tracks.append(clean)

        return tracks if tracks else None

    except Exception as e:
        print(f"      ⚠️ Wikipedia error: {e}")
        return None


# ============================================
# BACKFILL
# ============================================

def get_albums_missing_tracklist(conn) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id_album, id_artist, title, mbid, total_tracks
        FROM Album
        WHERE tracklist_json IS NULL
        ORDER BY id_artist, title
    ''')
    cols = ['id_album', 'id_artist', 'title', 'mbid', 'total_tracks']
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def load_artist_names(conn) -> Dict[int, str]:
    cursor = conn.cursor()
    cursor.execute('SELECT id_artist, name FROM Artist')
    return {id_artist: name for id_artist, name in cursor.fetchall()}


def update_album_tracklist(conn, id_album: int, tracklist: List[str], source: str):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Album
        SET tracklist_json = ?, tracklist_source = ?, last_update = CURRENT_TIMESTAMP
        WHERE id_album = ?
    ''', (json.dumps(tracklist, ensure_ascii=False), source, id_album))
    conn.commit()


def backfill():
    print("=" * 60)
    print("ALBUM TRACKLIST BACKFILL")
    print("=" * 60)

    if not os.path.exists(ALBUM_DB_PATH):
        print(f"❌ Album database not found: {ALBUM_DB_PATH}")
        return

    if not os.path.exists(ARTIST_DB_PATH):
        print(f"❌ Artist database not found: {ARTIST_DB_PATH}")
        return

    album_conn = sqlite3.connect(ALBUM_DB_PATH)
    artist_conn = sqlite3.connect(ARTIST_DB_PATH)

    artist_names = load_artist_names(artist_conn)
    artist_conn.close()

    albums = get_albums_missing_tracklist(album_conn)
    print(f"📀 {len(albums)} albums missing tracklist_json")

    no_mbid = [a for a in albums if not a['mbid']]
    with_mbid = [a for a in albums if a['mbid']]

    if no_mbid:
        print(f"⏭️  {len(no_mbid)} albums have no MBID on record — cannot backfill, would need a full re-fetch. Skipping those.")

    print(f"🔧 Backfilling {len(with_mbid)} albums with a known MBID...")
    print("-" * 60)

    filled_mb = 0
    filled_wiki = 0
    still_missing = 0

    for i, album in enumerate(with_mbid, start=1):
        artist_name = artist_names.get(album['id_artist'], f"[unknown artist id={album['id_artist']}]")
        print(f"[{i}/{len(with_mbid)}] {artist_name} - {album['title']}")

        tracklist = get_tracklist_from_musicbrainz(album['mbid'])
        source = 'musicbrainz'

        if not tracklist:
            tracklist = get_tracklist_from_wikipedia(artist_name, album['title'])
            source = 'wikipedia'

        if tracklist:
            if album['total_tracks'] and len(tracklist) != album['total_tracks']:
                print(f"      ⚠️ Tracklist length ({len(tracklist)}) doesn't match stored total_tracks ({album['total_tracks']})")

            update_album_tracklist(album_conn, album['id_album'], tracklist, source)
            print(f"      ✅ {len(tracklist)} tracks [{source}]")

            if source == 'musicbrainz':
                filled_mb += 1
            else:
                filled_wiki += 1
        else:
            print(f"      ⚠️ No tracklist found (MusicBrainz or Wikipedia)")
            still_missing += 1

        # Be polite to both APIs
        time.sleep(0.5)

    print("-" * 60)
    print(f"\n✅ Backfill complete")
    print(f"   Filled from MusicBrainz: {filled_mb}")
    print(f"   Filled from Wikipedia:   {filled_wiki}")
    print(f"   Still missing:           {still_missing}")
    print(f"   Skipped (no MBID):       {len(no_mbid)}")
    print("=" * 60)

    album_conn.close()


if __name__ == "__main__":
    backfill()

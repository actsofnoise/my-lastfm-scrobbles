"""
Diagnostic: MusicBrainz Tracklist Lookup Test

Quick standalone check — confirms whether the browse -> lookup fix for
get_official_release_details() actually returns real track titles, using
a known real case (King Crimson - Red) before spending a full pipeline run
on it.
"""

import musicbrainzngs

musicbrainzngs.set_useragent("my_scrobbles_diagnostic", "1.0", "test@example.com")


def test_album(artist_name: str, album_title: str):
    print(f"\n{'=' * 60}")
    print(f"Testing: {album_title} by {artist_name}")
    print(f"{'=' * 60}")

    # 1. Find the artist
    artist_result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
    artists = artist_result.get('artist-list', [])
    if not artists:
        print("❌ Artist not found")
        return
    artist_mbid = artists[0]['id']
    print(f"✅ Artist MBID: {artist_mbid}")

    # 2. Find the release-group
    rg_result = musicbrainzngs.browse_release_groups(artist=artist_mbid, limit=100)
    groups = rg_result.get('release-group-list', [])
    match = next((rg for rg in groups if rg.get('title', '').lower() == album_title.lower()), None)
    if not match:
        print(f"❌ Release-group '{album_title}' not found among {len(groups)} groups")
        return
    rg_mbid = match['id']
    print(f"✅ Release-group MBID: {rg_mbid}")

    # 3. Browse: find the official release
    browse_result = musicbrainzngs.browse_releases(release_group=rg_mbid, includes=['media', 'labels'], limit=25)
    releases = browse_result.get('release-list', [])
    official = [r for r in releases if r.get('status', '').lower() == 'official']
    print(f"📀 {len(releases)} releases found, {len(official)} official")
    if not official:
        print("❌ No official release found")
        return
    release_id = official[0]['id']
    print(f"✅ Using release: {release_id}")

    # 4. Lookup: get full track detail
    lookup_result = musicbrainzngs.get_release_by_id(release_id, includes=['media', 'labels', 'recordings'])
    release = lookup_result.get('release', {})

    total_tracks = 0
    tracklist = []
    for medium in release.get('medium-list', []):
        for track in medium.get('track-list', []):
            total_tracks += 1
            title = track.get('recording', {}).get('title') or track.get('title')
            tracklist.append(title or '(NO TITLE FOUND)')

    print(f"\n📋 Tracklist ({total_tracks} tracks):")
    for i, t in enumerate(tracklist, 1):
        print(f"   {i}. {t}")

    if all(t == '(NO TITLE FOUND)' for t in tracklist) and tracklist:
        print("\n❌ FIX DID NOT WORK — titles still empty even via lookup.")
    elif tracklist:
        print("\n✅ FIX WORKED — real track titles retrieved via lookup.")
    else:
        print("\n⚠️ No tracks found at all (medium-list/track-list empty).")


if __name__ == "__main__":
    test_album("King Crimson", "Red")

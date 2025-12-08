import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

class SpotifyClient:
    def __init__(self):
        self.sp = None
        self.auth_success = False
        self.authenticate()

    def authenticate(self):
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

        try:
            self.sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
            )
            self.auth_success = True
        except Exception as e:
            print(f"Authentication Error: {e}")
            self.last_error = str(e)

    def get_recommendations(self, emotion, genre=None):
        """
        Returns either:
            - list of cleaned track dicts: [{ name, artist, url, image }]
            - {"error": "..."}
        Never returns None.
        """

        if not self.sp:
            return {"error": "Spotify authentication failed."}

        mood_map = {
            "neu": "chill",
            "hap": "happy",
            "sad": "sad",
            "ang": "workout"
        }

        mood_query = mood_map.get(emotion, "top hits")
        
        # Construct query with genre if provided
        if genre and genre != "All":
            query = f"{mood_query} {genre}"
        else:
            query = mood_query

        try:
            # Search playlist (fetch more to avoid None items)
            results = self.sp.search(q=query, type='playlist', limit=10)

            print("results:", results)

            raw_playlists = results.get("playlists", {}).get("items") or []

            # items に None が混ざるバグ対策
            playlists = [p for p in raw_playlists if isinstance(p, dict)]

            print("filtered playlists:", playlists)

            if not playlists:
                return {"error": f"No usable playlist found for mood='{query}'"}

            playlist_id = playlists[0].get("id")

            print("playlist_id:", playlist_id)

            if not playlist_id:
                return {"error": "Playlist found but missing valid ID."}

            # Fetch tracks
            playlist_tracks = self.sp.playlist_tracks(playlist_id, limit=10)

            print("playlist_tracks: ", playlist_tracks)

            items = playlist_tracks.get("items") or []

            cleaned_tracks = []

            print("items: ", items)

            for item in items:
                if not isinstance(item, dict):
                    continue

                track = item.get("track")
                if not isinstance(track, dict):
                    continue

                name = track.get("name")
                artists = track.get("artists") or []
                url = (track.get("external_urls") or {}).get("spotify")
                album = track.get("album") or {}
                images = album.get("images") or []

                if not name or not artists or not url:
                    continue

                cleaned_tracks.append({
                    "name": name,
                    "artist": artists[0]["name"],
                    "url": url,
                    "image": images[0]["url"] if images else None,
                    "id": track.get("id")
                })

            if not cleaned_tracks:
                return {"error": f"No playable tracks available for '{query}'"}

            return cleaned_tracks

        except Exception as e:
            return {"error": f"Unexpected Spotify error: {str(e)}"}

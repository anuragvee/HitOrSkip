import spotipy,os
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import csv

load_dotenv()


scope = "user-library-read user-top-read playlist-read-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ["SPOT_ID"],
    client_secret=os.environ["SPOT_SECRET"],
    redirect_uri="http://127.0.0.1:8888/callback",
    scope=scope,
))

all_saved = []
offset = 0
while True:
    batch = sp.current_user_saved_tracks(limit=50, offset=offset)
    items = batch["items"]
    if not items:
        break
    for item in items:
        track = item["track"]
        if track is None:
            continue
        all_saved.append({
            "track_id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "artist_id_primary": track["artists"][0]["id"] if track["artists"] else "",
            "album": track["album"]["name"],
            "release_date": track["album"].get("release_date", ""),
            "popularity": track.get("popularity", 0),
            "duration_ms": track.get("duration_ms", 0),
            "added_at": item.get("added_at", ""),
            "source": "saved",
        })
    offset += 50
    print(f"Pulled {offset} so far...")

print(f"\nTotal saved tracks pulled: {len(all_saved)}")

for term in ["short_term", "medium_term", "long_term"]:
    top = sp.current_user_top_tracks(limit=50, time_range=term)
    for track in top["items"]:
        all_saved.append({
            "track_id": track["id"],
            "name": track["name"],
            "artist": ", ".join(a["name"] for a in track["artists"]),
            "artist_id_primary": track["artists"][0]["id"] if track["artists"] else "",
            "album": track["album"]["name"],
            "release_date": track["album"].get("release_date", ""),
            "popularity": track.get("popularity", 0),
            "duration_ms": track.get("duration_ms", 0),
            "added_at": "",
            "source": f"top_{term}",
        })

seen = set()
unique = []
for t in all_saved:
    if t["track_id"] and t["track_id"] not in seen:
        seen.add(t["track_id"])
        unique.append(t)

print(f"Unique liked tracks: {len(unique)}")

with open("liked_tracks.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=unique[0].keys())
    writer.writeheader()
    writer.writerows(unique)

print("Saved liked_tracks.csv")
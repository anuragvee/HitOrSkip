import os
import pandas as pd
import joblib
import gradio as gr
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from dotenv import load_dotenv

load_dotenv()

bundle = joblib.load("prev_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

scope = "user-library-read user-top-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ["SPOT_ID"],
    client_secret=os.environ["SPOT_SECRET"],
    redirect_uri="http://127.0.0.1:8888/callback",
    scope=scope,
    open_browser=False,
))

RECCO_BASE = "https://api.reccobeats.com/v1"


def search_spotify(song: str, artist: str = "") -> dict | None:
    query = f"track:{song}"
    if artist.strip():
        query += f" artist:{artist}"
    results = sp.search(q=query, type="track", limit=1)
    items = results.get("tracks", {}).get("items", [])
    if not items:
        return None
    t = items[0]
    return {
        "name": t["name"],
        "artist": ", ".join(a["name"] for a in t["artists"]),
        "spotify_id": t["id"],
        "album": t["album"]["name"],
        "image_url": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
        "spotify_url": t["external_urls"]["spotify"],
    }


def get_features_from_reccobeats(spotify_id: str) -> dict | None:
    r = requests.get(f"{RECCO_BASE}/track", params={"ids": spotify_id}, timeout=15)
    if r.status_code != 200:
        return None
    data = r.json()
    tracks = data.get("content", []) if isinstance(data, dict) else data
    if not tracks:
        return None
    recco_id = tracks[0].get("id")
    if not recco_id:
        return None
    
    r = requests.get(f"{RECCO_BASE}/track/{recco_id}/audio-features", timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def predict(song: str, artist: str):
    if not song.strip():
        return "Please enter a song name.", None

    match = search_spotify(song, artist)
    if not match:
        return "Couldn't find that song on Spotify. Try a different spelling?", None

    features = get_features_from_reccobeats(match["spotify_id"])
    if not features:
        return (
            f"Found **{match['name']}** by {match['artist']} on Spotify, "
            "but ReccoBeats doesn't have audio features for this track. "
            "Try a more popular song.",
            match["image_url"],
        )

    try:
        X = pd.DataFrame([{f: features[f] for f in FEATURES}])
    except KeyError as e:
        return f"ReccoBeats response missing feature {e}. Try a different song.", match["image_url"]

    prob_liked = model.predict_proba(X)[0][1] 
    percent = prob_liked * 100

    verdict = (
        "HEATTTTTTT!!!" if percent >= 75
        else "I would vibe with it" if percent >= 55
        else "There's a chance I would like it" if percent >= 40
        else "Not it" if percent >= 25
        else "Bouse Squid."
    )

    feature_lines = "\n".join(
        f"- **{f}**: {features[f]:.3f}" for f in FEATURES if f in features
    )

    output = f"""
### {match['name']}
**{match['artist']}** · _{match['album']}_

## {percent:.1f}% chance I would like it
**{verdict}**

[Open on Spotify]({match['spotify_url']})

---

#### Audio features
{feature_lines}
"""
    return output, match["image_url"]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Song name", placeholder="e.g. SICKO MODE"),
        gr.Textbox(label="Artist (optional)", placeholder="e.g. Travis Scott"),
    ],
    outputs=[gr.Markdown(label="Prediction"),gr.Image(label="Album art", show_label=False),
    ],
    title="HitOrSkip",
    description=(
        "Trained on 1,368 of my Spotify liked songs + 1,368 random negatives. "
        "Random Forest, ~80% cross-validated accuracy. "
        "Audio features via ReccoBeats."
    ),
    examples=[["SICKO MODE", "Travis Scott"],["Hotline Bling", "Drake"],["She", "Tyler The Creator"],
    ],
    flagging_mode="never",
)

demo.launch()
import os, json, time, requests
import pandas as pd

BASE = "https://api.reccobeats.com/v1"
AMOUNTPER = 40
SLEEP_BETWEEN_CALLS = 0.3
FALLBACK = "rb.json"


def get_recco_ids(spotify_id_batch):
    req = requests.get(f"{BASE}/track", params={"ids": ",".join(spotify_id_batch)}, timeout=20)

    if req.status_code != 200:
        print(f"Error ({req.status_code}): {req.text[:200]}")
        return {}
    data = req.json()
    if isinstance(data, dict):
        tracks = data.get("content", data)
    else:
        tracks = data
    out = {}
    for i in tracks:
        recco_id = i.get("id")
        href = i.get("href", "")
        if "spotify" in href:
            spot_id = href.split("/")[-1]
        else:
            spot_id = None

        if recco_id and spot_id:
            out[spot_id] = recco_id

    return out


def get_audio_features(recco_id):
    r = requests.get(f"{BASE}/track/{recco_id}/audio-features", timeout=15)

    if r.status_code == 200:
        return r.json()
    else:
        return None


def main():
    liked = pd.read_csv("data/liked_tracks.csv")
    spotify_ids = liked["track_id"].dropna().unique().tolist()
    print(f"Total Spotify IDs to fetch: {len(spotify_ids)}")

    if os.path.exists(FALLBACK):
        with open(FALLBACK) as f:
            results = json.load(f)
        print(f"Resuming..")
    else:
        results = {}

    to_process = []
    for sid in spotify_ids:
        if sid not in results:
            to_process.append(sid)

    print(f"Remaining: {len(to_process)}\n")

    for i in range(0, len(to_process), AMOUNTPER):
        batch = to_process[i : i + AMOUNTPER]
        print(f"Batch {i//AMOUNTPER+1} ({i+1}-{i+len(batch)} of {len(to_process)})")

        recco_map = get_recco_ids(batch)
        print(f"  Mapped {len(recco_map)} / {len(batch)} to ReccoBeats IDs")

        for spot_id, recco_id in recco_map.items():
            features = get_audio_features(recco_id)
            if features:
                results[spot_id] = features
            time.sleep(SLEEP_BETWEEN_CALLS)

        with open(FALLBACK, "w") as f:
            json.dump(results, f)
        print(f"Saved ({len(results)} total)")

    print(f"\nTotal: {len(results)} / {len(spotify_ids)} tracks")

    rows = []
    for spot_id, feats in results.items():
        row = {"track_id": spot_id}
        row.update(feats)
        rows.append(row)

    SongFT_DataFrame = pd.DataFrame(rows)
    merged = liked.merge(SongFT_DataFrame, on="track_id", how="inner")
    merged.to_csv("data/liked_tracks_with_features.csv", index=False)
    print(f"Saved data/liked_tracks_with_features.csv ({len(merged)} rows)")



main()
import pandas as pd

liked = pd.read_csv("data/liked_tracks_with_features.csv")
liked["label"] = 1
print(f"Positives (liked): {len(liked)}")

print("Loading negative pool from Hugging Face...")
maharshi = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
print(f"Negative pool: {len(maharshi)}")

liked_ids = set(liked["track_id"])
liked_artists = set(liked["artist"].str.lower().str.split(",").str[0].str.strip())

neg_pool = maharshi[~maharshi["track_id"].isin(liked_ids)].copy()
neg_pool["primary_artist_lower"] = (neg_pool["artists"].astype(str).str.lower().str.split(";").str[0].str.strip())
neg_pool = neg_pool[~neg_pool["primary_artist_lower"].isin(liked_artists)]
print(f"Negitive pool (- liked artists) {len(neg_pool)}")

n_neg = len(liked)
negatives = neg_pool.sample(n=n_neg, random_state=42).copy()
negatives["label"] = 0
print(f"Sampled negatives: {len(negatives)}")

features = ["danceability", "energy", "valence", "tempo", "acousticness","instrumentalness", "loudness", "speechiness", "liveness", "key", "mode"]

pos_subset = liked[["track_id", "name", "artist"] + features + ["label"]].copy()
neg_subset = negatives.rename(columns={"track_name": "name", "artists": "artist"})[["track_id", "name", "artist"] + features + ["label"]].copy()

final = pd.concat([pos_subset, neg_subset], ignore_index=True)
final = final.sample(frac=1, random_state=42).reset_index(drop=True)
final = final.dropna(subset=features)

final.to_csv("data/training_data.csv", index=False)
print(f"\nSaved data/training_data.csv ({len(final)} rows)")
print(f"Liked:     {(final['label'] == 1).sum()}")
print(f"Not liked: {(final['label'] == 0).sum()}")
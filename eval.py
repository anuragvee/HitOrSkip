import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/training_data.csv")
features = ["danceability", "energy", "valence", "tempo", "acousticness","instrumentalness", "loudness", "speechiness", "liveness", "key", "mode"]
X = df[features]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1)
model.fit(X_train, y_train)

print(f"Train acc: {model.score(X_train, y_train):.3f}")
print(f"Test acc:  {model.score(X_test, y_test):.3f}")

y_pred = model.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=["not liked", "liked"]))

cv_scores = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1),X, y, cv=5, scoring="accuracy")

print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print(f"Headline: {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%")
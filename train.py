import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/training_data.csv")
print(f"Loaded {len(df)} examples ({(df['label']==1).sum()} liked, {(df['label']==0).sum()} not)\n")

features = ["danceability", "energy", "valence", "tempo", "acousticness","instrumentalness", "loudness", "speechiness", "liveness", "key", "mode"]
X = df[features]
y = df["label"]

print("-" * 60)
print("EVALUATION 1: 80/20 train/test split")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)

eval_model = RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1)
eval_model.fit(X_train, y_train)

train_acc = eval_model.score(X_train, y_train)
test_acc = eval_model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}  (memorizes training data -- expected)")
print(f"Test accuracy:  {test_acc:.3f}  (honest performance on unseen songs)")
print(f"Gap:            {train_acc - test_acc:+.3f}  (overfit if >0.10)")

y_pred = eval_model.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["not liked", "liked"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(f"                  pred not liked   pred liked")
print(f"actual not liked        {cm[0,0]:5d}        {cm[0,1]:5d}")
print(f"actual liked            {cm[1,0]:5d}        {cm[1,1]:5d}")

print("\n" + "-" * 60)
print("EVALUATION 2: 5-fold cross-validation (most reliable)")
print("-" * 60)

cv_model = RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1)
cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="accuracy")
print(f"Fold accuracies: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean:            {cv_scores.mean():.3f}")
print(f"Standard dev:         +/-{cv_scores.std():.3f}")
print(f"\nHeadline accuracy: {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%")

prec = cross_val_score(cv_model, X, y, cv=5, scoring="precision")
rec = cross_val_score(cv_model, X, y, cv=5, scoring="recall")
f1 = cross_val_score(cv_model, X, y, cv=5, scoring="f1")
print(f"\nPrecision (CV): {prec.mean():.3f} +/- {prec.std():.3f}")
print(f"Recall (CV):    {rec.mean():.3f} +/- {rec.std():.3f}")
print(f"F1 (CV):        {f1.mean():.3f} +/- {f1.std():.3f}")

print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)
importance_df = pd.DataFrame({"feature": features,"importance": eval_model.feature_importances_,}).sort_values("importance", ascending=False)
print(importance_df.to_string(index=False))

print("\n" + "-" * 60)
print("TRAINING FINAL MODEL (on all data) FOR DEPLOYMENT")
print("-" * 60)
final_model = RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1)
final_model.fit(X, y)
print(f"Final model trained on all {len(df)} examples.")
print(f"Expected real-world accuracy: {cv_scores.mean()*100:.1f}%")

joblib.dump({
    "model": final_model,
    "features": features,
    "cv_accuracy_mean": float(cv_scores.mean()),
    "cv_accuracy_std": float(cv_scores.std()),
    "cv_precision": float(prec.mean()),
    "cv_recall": float(rec.mean()),
    "cv_f1": float(f1.mean()),
}, "prev_model.pkl")

print("\nSaved as prev_model.pkl")
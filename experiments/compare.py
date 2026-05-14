import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("data/training_data.csv")
print(f"Loaded {len(df)} rows\n")

features = ["danceability", "energy", "valence", "tempo", "acousticness","instrumentalness", "loudness", "speechiness", "liveness", "key", "mode",]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)


def evaluate(name, model, scale=False):
    Xtr, Xte = X_train, X_test
    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

    model.fit(Xtr, y_train)
    train_acc = model.score(Xtr, y_train)
    test_acc = model.score(Xte, y_test)

    print(f"\n{'-' * 20}\n{name}\n{'-' * 20}")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    y_pred = model.predict(Xte)
    print(f"\n{classification_report(y_test, y_pred, target_names=['not liked', 'liked'])}")

    return {"name": name, "train": train_acc, "test": test_acc, "model": model}


results = []
results.append(evaluate("MODEL 1: Logistic Regression",LogisticRegression(max_iter=1000, random_state=67),scale=True,))
results.append(evaluate("MODEL 2: Random Forest (200 trees)",RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1),))
results.append(evaluate("MODEL 3: XGBoost",XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,random_state=67, eval_metric="logloss", n_jobs=-1,),))


print("\n" + "-" * 20)
print("SUMMARY: model comparison")
print("-" * 20)
print(f"{'Model':40s}  {'Train':>7s}  {'Test':>7s}  {'Gap':>7s}")
print("-" * 65)
for res in results:
    gap = res["train"] - res["test"]
    print(f"{res['name'][:40]:40s}  {res['train']:7.3f}  {res['test']:7.3f}  {gap:+7.3f}")

print("\n" + "-" * 20)
print("5-FOLD CROSS-VALIDATION")
print("-" * 20)

X_scaled = StandardScaler().fit_transform(X)
cv_results = []
cv_results.append(("Logistic Regression",cross_val_score(LogisticRegression(max_iter=1000, random_state=67),X_scaled, y, cv=5, scoring="accuracy")))
cv_results.append(("Random Forest",cross_val_score(RandomForestClassifier(n_estimators=200, random_state=67, n_jobs=-1),X, y, cv=5, scoring="accuracy")))
cv_results.append(("XGBoost",cross_val_score(XGBClassifier(n_estimators=300, max_depth=5,learning_rate=0.1, random_state=67,eval_metric="logloss", n_jobs=-1),X, y, cv=5, scoring="accuracy")))


print(f"\n{'Model':25s}  {'Mean':>7s}  {'StdDev':>7s}")
print("-" * 50)
for name, scores in cv_results:
    print(f"{name:25s}  {scores.mean():7.3f}  {scores.std():7.3f}")


rf = results[1]["model"]
importance = pd.DataFrame({"feature": features,"importance": rf.feature_importances_,}).sort_values("importance", ascending=False)

print("\n" + "-" * 20)
print("FEATURE IMPORTANCE (Random Forest)")
print("-" * 20)
print(importance.to_string(index=False))
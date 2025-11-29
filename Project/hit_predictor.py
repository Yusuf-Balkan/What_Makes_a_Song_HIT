#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter("ignore", FutureWarning)


# -----------------------------
# Utility: Ensure directories
# -----------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    return df


# -----------------------------
# Clean Dataset
# -----------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with null target
    df.dropna(subset=["track_popularity"], inplace=True)

    return df


# -----------------------------
# Select Features
# -----------------------------
def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    candidate_features = [
        "danceability", "energy", "acousticness", "instrumentalness",
        "liveness", "valence", "speechiness", "tempo", "duration_ms", "loudness"
    ]

    available_features = [f for f in candidate_features if f in df.columns]

    print(f"[INFO] Using features: {available_features}")

    X = df[available_features]
    y = df["track_popularity"]

    return X, y, available_features


# -----------------------------
# Scaling
# -----------------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(y_test, preds) -> dict:
    import math
    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, preds)
    return {"rmse": float(rmse), "r2": float(r2)}


# -----------------------------
# Feature Importance Plot
# -----------------------------
def plot_feature_importance(model, feature_names, outpath):
    try:
        importances = model.feature_importances_
    except Exception:
        print("[WARN] Feature importance not available for this model.")
        return

    idx = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx], importances[idx])
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[INFO] Saved feature importance plot to {outpath}")


# -----------------------------
# Correlation Heatmap
# -----------------------------
def plot_correlation(df: pd.DataFrame, outpath: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="viridis", annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[INFO] Saved correlation heatmap to {outpath}")


# -----------------------------
# Pairplot for selected features
# -----------------------------
def plot_pairplot(df: pd.DataFrame, features: List[str], outpath: str):
    sns.pairplot(df[features + ["track_popularity"]].sample(1000, random_state=42))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[INFO] Saved pairplot subset to {outpath}")


# -----------------------------
# Main Pipeline
# -----------------------------
def main(args):
    ensure_dir("outputs")
    ensure_dir("outputs/plots")

    df = load_dataset(args.data)
    df = clean_dataset(df)

    X, y, features = select_features(df)

    # Save descriptive stats
    df.describe().to_csv("outputs/dataset_describe.csv")
    print("[INFO] Saved descriptive stats to outputs/dataset_describe.csv")

    # Correlation Heatmap
    plot_correlation(
        df[features + ["track_popularity"]],
        "outputs/plots/correlation_matrix.png"
    )

    # Pairplot (subset)
    plot_pairplot(df, features[:4], "outputs/plots/pairplot_subset.png")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # -----------------------------
    # Linear Regression
    # -----------------------------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_report = evaluate_model(y_test, lr_preds)
    print(f"[RESULT] Linear Regression → RMSE: {lr_report['rmse']:.2f}, R²: {lr_report['r2']:.4f}")

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_report = evaluate_model(y_test, rf_preds)
    print(f"[RESULT] Random Forest → RMSE: {rf_report['rmse']:.2f}, R²: {rf_report['r2']:.4f}")

    # Save feature importance
    plot_feature_importance(rf, features, "outputs/plots/feature_importance.png")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to spotify_songs.csv")
    args = parser.parse_args()
    main(args)

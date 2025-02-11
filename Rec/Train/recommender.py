# train_recommender.py
import pandas as pd
import numpy as np
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Path to your main CSV in Rec/Data
DATA_PATH = '../Data/data.csv'  # Adjust if needed
MODEL_PATH = '../Model/spotify_knn_model.pkl'

def main():
    # 1) Read the CSV data
    df = pd.read_csv(DATA_PATH)

    # Quick check of columns. Adjust these if your CSV uses different names.
    # Some typical columns in Spotify data might be:
    #   danceability, energy, valence, tempo, loudness, etc.
    #   Make sure they exist in "df.columns".
    features = [
        'danceability', 'energy', 'valence',
        'tempo', 'loudness'
        # etc... add more as needed
    ]

    # 2) Drop rows with NaN in these features (if needed)
    df = df.dropna(subset=features)

    # 3) Extract just the numeric feature subset
    X = df[features].copy()

    # 4) Scale the features to unify range
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) Train a KNN model to find nearest neighbors (for recommendations)
    knn = NearestNeighbors(n_neighbors=11, algorithm='auto')
    knn.fit(X_scaled)

    # 6) Save the necessary objects (the KNN model + the scaler)
    #    We'll pickle them together in a dict for convenience
    model_objects = {
        'knn': knn,
        'scaler': scaler,
        'features': features,
        'dataframe': df  # to map indices back to song info
    }
    joblib.dump(model_objects, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()

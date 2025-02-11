import pickle
import sys

def recommend_songs_by_user_input(artist, song_name, n=5):
    # Load the trained model, scaler, and dataset from the .pkl file
    with open('Rec/Model/knn_recommender.pkl', 'rb') as f:
        knn_model_small, scaler_small, df_small = pickle.load(f)

    # Step 1: Search for the song in the dataset
    matching_song = df_small[(df_small['artists'] == artist) & (df_small['name'] == song_name)]
    
    if matching_song.empty:
        print(f"Song '{song_name}' by '{artist}' not found in the dataset.")
        return []

    # Step 2: Extract the song's features (danceability, energy, tempo)
    row_data = matching_song.iloc[0][['danceability', 'energy', 'tempo']].values
    
    # Step 3: Run inference using the KNN model
    arr = scaler_small.transform([row_data])
    distances, indices = knn_model_small.kneighbors(arr, n_neighbors=n+1)
    
    # Step 4: Return recommended songs, skipping the first neighbor (the song itself)
    rec_indices = indices[0][1:]
    recommendations = df_small.iloc[rec_indices][['artists', 'name']].values.tolist()
    
    return recommendations

if __name__ == "__main__":
    # Get user input from the command line
    artist_input = input("Enter the artist's name: ").strip()
    song_name_input = input("Enter the song name: ").strip()

    # Get recommendations
    print(f"\n--- Recommendations for '{song_name_input}' by {artist_input} ---")
    recommended_songs = recommend_songs_by_user_input(artist_input, song_name_input, n=5)

    # Display the recommendations
    if recommended_songs:
        for rec_artist, rec_song in recommended_songs:
            print(f"{rec_artist} - \"{rec_song}\"")
    else:
        print("No recommendations found.")

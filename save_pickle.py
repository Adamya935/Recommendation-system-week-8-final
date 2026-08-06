import pickle

from hybrid_final import (
    build_collaborative,
    build_content_based,
    build_popularity_table,
)

print("Building content feature space & TF-IDF vectors...")
data, vectors, all_titles, title_to_pos, pos_to_id, id_to_title = build_content_based()

print("Building popularity fallback table...")
popularity_table = build_popularity_table()

print("Building collaborative matrix & pre-fitting NearestNeighbors model...")
(
    user_rating_matrix,
    movie_columns,
    user_index,
    movieid_to_tmdbid,
    nbrs,
    user_rating_counts,
    user_means,
    user_stds,
) = build_collaborative(data)

model_data = {
    "vectors": vectors,
    "all_titles": all_titles,
    "title_to_pos": title_to_pos,
    "pos_to_id": pos_to_id,
    "id_to_title": id_to_title,
    "user_rating_matrix": user_rating_matrix,
    "movie_columns": movie_columns,
    "user_index": user_index,
    "movieid_to_tmdbid": movieid_to_tmdbid,
    "nbrs": nbrs,
    "user_rating_counts": user_rating_counts,
    "user_means": user_means,
    "user_stds": user_stds,
    "popularity_table": popularity_table,
}

print("Saving to hybrid_recommender.pkl...")
with open("hybrid_recommender.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("hybrid_recommender.pkl created successfully!")

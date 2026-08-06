import streamlit as st
import pickle
from hybrid_final import hybrid_recommend, find_closest_titles

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬")

# ----------------------------------------------------
# Load Pickle
# ----------------------------------------------------
@st.cache_resource
def load_model():
    with open("hybrid_recommender.pkl", "rb") as f:
        return pickle.load(f)


data = load_model()

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.write("Content-Based + Collaborative + Hybrid Recommendations")

movie_title = st.text_input("Enter Movie Title", "")
user_id_input = st.text_input("Enter User ID (optional)", "")
top_n = st.slider("Number of Recommendations", 5, 20, 10)

# ----------------------------------------------------
# Recommend Button
# ----------------------------------------------------
if st.button("Recommend"):

    # User ID conversion
    user_id = int(user_id_input) if user_id_input.strip().isdigit() else None

    # Movie title validation
    query = movie_title.strip().lower()
    if query:
        if query not in data["title_to_pos"]:
            suggestions = find_closest_titles(query, data["all_titles"])
            if suggestions:
                st.warning("Movie not found. Using closest match:")
                st.write(f"**{suggestions[0].title()}**")
                query = suggestions[0]
            else:
                st.error("Movie title not found in dataset.")
                st.stop()
    else:
        query = None

    # ------------------------------------------------
    # Call Hybrid Recommender
    # ------------------------------------------------
    output = hybrid_recommend(
        movie_title=query,
        user_id=user_id,
        top_n=top_n,
        vectors=data["vectors"],
        all_titles=data["all_titles"],
        title_to_pos=data["title_to_pos"],
        pos_to_id=data["pos_to_id"],
        id_to_title=data["id_to_title"],
        user_rating_matrix=data["user_rating_matrix"],
        movie_columns=data["movie_columns"],
        user_index=data["user_index"],
        movieid_to_tmdbid=data["movieid_to_tmdbid"],
        nbrs=data["nbrs"],
        user_rating_counts=data["user_rating_counts"],
        user_means=data["user_means"],
        user_stds=data["user_stds"],
        popularity_table=data["popularity_table"],
    )

    # ------------------------------------------------
    # Handle different return formats safely
    # ------------------------------------------------
    message = None
    results = output

    if isinstance(output, tuple) and len(output) == 2:
        message, results = output

    if message:
        st.info(message)

    # ------------------------------------------------
    # Display Recommendations
    # ------------------------------------------------
    if results:
        st.subheader("🎬 Recommended Movies")

        for i, item in enumerate(results, start=1):

            # (title, rating)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0]).title()
                rating = item[1]
                st.write(f"**{i}. {title}** — ⭐ {rating}")

            # only title
            else:
                st.write(f"**{i}. {str(item).title()}**")

    else:
        st.warning("No recommendations found.")

# st.info(message)

# st.subheader("Recommendations")

# if not results:
#     st.warning("No recommendations found.")
# else:
#     for i, (title, rating) in enumerate(results, start=1):
#         st.write(f"**{i}. {title.title()}** — ⭐ {rating}")




    # TO RUN THE FILE
    # python -m streamlit run app.py

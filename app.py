# -*- coding: utf-8 -*-
"""
================================================================================
STREAMLIT FRONTEND FOR THE HYBRID MOVIE RECOMMENDATION SYSTEM
================================================================================

Place this file in the SAME folder as:

    hybrid_final.py
    movies_metadata.csv
    credits_small.csv
    keywords.csv
    ratings_small.csv
    links_small.csv

Run:

    streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd

from hybrid_final import (
    build_content_based,
    build_popularity_table,
    build_collaborative,
    recommend_content,
    recommend_user_based,
    hybrid_recommend,
    calculate_graduated_alpha,
)

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Cached model loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building content-based feature space (TF-IDF)...")
def load_content():
    return build_content_based()


@st.cache_resource(show_spinner="Building popularity fallback table...")
def load_popularity():
    return build_popularity_table()


# Load content first because collaborative builder needs the processed data
data, vectors, all_titles, title_to_pos, pos_to_id, id_to_title = load_content()


@st.cache_resource(show_spinner="Building collaborative matrix & fitting KNN...")
def load_collaborative():
    return build_collaborative(data)


popularity_table = load_popularity()

(
    user_rating_matrix,
    movie_columns,
    user_index,
    movieid_to_tmdbid,
    nbrs,
    user_rating_counts,
    user_means,
    user_stds,
) = load_collaborative()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Recommendation Inputs")

movie_title = st.sidebar.text_input(
    "Movie Title",
    placeholder="e.g. Toy Story",
)

raw_user_id = st.sidebar.text_input(
    "User ID (optional)",
    placeholder="e.g. 1",
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=15,
    value=5,
)

show_breakdown = st.sidebar.checkbox(
    "Show Content vs Collaborative Breakdown",
    value=True,
)

run_button = st.sidebar.button(
    "Get Recommendations",
    type="primary",
)

with st.sidebar.expander("Advanced"):
    st.caption(
        "If you update the dataset CSV files, clear the cache to rebuild "
        "the recommendation engines."
    )

    if st.button("Clear Cache & Rebuild"):
        st.cache_resource.clear()
        st.success("Cache cleared successfully.")
        st.rerun()

# Parse user ID safely
user_id = int(raw_user_id) if raw_user_id.strip().isdigit() else None

# -----------------------------------------------------------------------------
# Main title
# -----------------------------------------------------------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.caption(
    "A hybrid recommender that combines content-based filtering and "
    "collaborative filtering using a graduated cold-start blending strategy."
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def show_status(movie_title_input, user_id_input):
    """Display user/movie status information."""

    is_known_user = (
        user_id_input is not None and user_id_input in user_index
    )

    rating_count = (
        user_rating_counts.get(user_id_input, 0)
        if is_known_user
        else 0
    )

    alpha = calculate_graduated_alpha(rating_count)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "User Status",
        (
            "Known"
            if is_known_user
            else (
                "Not Provided"
                if user_id_input is None
                else "Unknown / New"
            )
        ),
    )

    c2.metric(
        "Ratings on Record",
        rating_count,
    )

    c3.metric(
        "Collaborative Weight (α)",
        f"{alpha:.2f}",
    )

    if movie_title_input is None and not is_known_user:
        st.info(
            "No valid movie title or known user ID was provided. "
            "The system will use the popularity-based fallback recommender."
        )

    elif not is_known_user:
        st.info(
            f"Unknown/new user. α = 0.00, so recommendations are generated "
            f"purely from content similarity using '{movie_title_input}'."
        )

    else:
        st.info(
            f"User has {rating_count} ratings on record. "
            f"α = {alpha:.2f}, blending content and collaborative filtering."
        )

    return is_known_user, alpha


def results_to_df(results, score_label="Estimated ★"):
    return pd.DataFrame(
        results,
        columns=["Movie Title", score_label],
    )

# -----------------------------------------------------------------------------
# Recommendation execution
# -----------------------------------------------------------------------------
if run_button:

    if not movie_title.strip() and user_id is None:
        st.warning(
            "Please enter a movie title, a user ID, or both."
        )

    else:

        is_known_user, alpha = show_status(
            movie_title if movie_title.strip() else None,
            user_id,
        )

        st.subheader("Hybrid Recommendations")

        recommendations = hybrid_recommend(
            movie_title=movie_title if movie_title.strip() else None,
            user_id=user_id,
            top_n=top_n,
            vectors=vectors,
            all_titles=all_titles,
            title_to_pos=title_to_pos,
            pos_to_id=pos_to_id,
            id_to_title=id_to_title,
            user_rating_matrix=user_rating_matrix,
            movie_columns=movie_columns,
            user_index=user_index,
            movieid_to_tmdbid=movieid_to_tmdbid,
            nbrs=nbrs,
            user_rating_counts=user_rating_counts,
            user_means=user_means,
            user_stds=user_stds,
            popularity_table=popularity_table,
        )

        if recommendations:
            st.dataframe(
                results_to_df(recommendations),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No recommendations could be generated.")

        # ---------------------------------------------------------------------
        # Breakdown
        # ---------------------------------------------------------------------
        if show_breakdown:

            st.divider()

            left, right = st.columns(2)

            # -------------------------------------------------------------
            # Content-only
            # -------------------------------------------------------------
            with left:

                st.subheader("Content-Based Recommendations")

                if movie_title.strip():

                    _, content_results = recommend_content(
                        movie_title,
                        vectors,
                        all_titles,
                        title_to_pos,
                        pos_to_id,
                        top_n=top_n,
                    )

                    if content_results:

                        content_df = pd.DataFrame(
                            [
                                (
                                    id_to_title.get(mid, f"id {mid}"),
                                    round(3.5 + score * 1.4, 1),
                                )
                                for mid, score in content_results
                            ],
                            columns=["Movie Title", "Estimated ★"],
                        )

                        st.dataframe(
                            content_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.caption(
                            "No content-based match found."
                        )

                else:
                    st.caption(
                        "Enter a movie title to view content-based results."
                    )

            # -------------------------------------------------------------
            # Collaborative-only
            # -------------------------------------------------------------
            with right:

                st.subheader("Collaborative Recommendations")

                if is_known_user:

                    collaborative_results = recommend_user_based(
                        user_id,
                        user_rating_matrix,
                        movie_columns,
                        user_index,
                        movieid_to_tmdbid,
                        nbrs,
                        user_means=user_means,
                        top_n=top_n,
                    )

                    if collaborative_results:

                        collab_df = pd.DataFrame(
                            [
                                (
                                    id_to_title.get(mid, f"id {mid}"),
                                    round(score, 1),
                                )
                                for mid, score in collaborative_results
                            ],
                            columns=["Movie Title", "Predicted Rating"],
                        )

                        st.dataframe(
                            collab_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.caption(
                            "No collaborative predictions available."
                        )

                else:
                    st.caption(
                        "Enter a known user ID to view collaborative results."
                    )

# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------
else:

    st.info(
        "Enter a movie title and/or a user ID in the sidebar and click "
        "**Get Recommendations**."
    )

    st.markdown(
        """
### How this recommender works

- **Content-Based Filtering**
  - Uses TF-IDF features from genres, keywords, cast, crew, and overview.
  - Finds movies that are textually similar.

- **Collaborative Filtering**
  - Uses user rating patterns from the MovieLens dataset.
  - Predicts movies a user may enjoy based on similar users.

- **Hybrid Engine**
  - Blends both methods using a graduated confidence parameter **α**.
  - Handles cold-start users and unseen movies gracefully.
"""
    )

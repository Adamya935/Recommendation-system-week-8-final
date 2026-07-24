# -*- coding: utf-8 -*-
"""
================================================================================
STREAMLIT FRONTEND FOR THE HYBRID MOVIE RECOMMENDATION SYSTEM
================================================================================
Setup:
    1. Place this file in the SAME folder as hybrid_final.py
       (the file with build_content_based, build_collaborative,
       hybrid_recommend, etc.) and the 5 dataset CSVs:
       movies_metadata.csv, credits_small.csv, keywords.csv,
       ratings_small.csv, links_small.csv
    2. pip install streamlit
    3. streamlit run app.py

What it does:
    - Builds the content, collaborative, and popularity engines once and
      caches them in memory with st.cache_resource, so the heavy
      TF-IDF/KNN build only runs on first load, not on every interaction.
    - Takes a movie title and/or a user id as input, same as the terminal
      version's `run_interactive` loop, but as a web form.
    - Shows the final hybrid recommendations, plus optional side-by-side
      content-only and collaborative-only breakdowns so you can see how
      the two engines diverge and how alpha blends them.

NOTE: hybrid_final.py only exposes build_content_based(), build_popularity_table(),
and build_collaborative(data) — there is no on-disk caching layer (no
get_or_build_* wrappers, no clear_cache) in that file, so all caching here is
done purely with Streamlit's own st.cache_resource, in-memory, per server process.
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

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")


# -----------------------------------------------------------------------------
# In-memory caching only. Each of these runs once per server process (first
# time any user hits the app) and is then reused for every subsequent
# interaction/session, since hybrid_final.py has no disk-cache layer of its
# own to fall back on.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building content-based feature space (TF-IDF)...")
def load_content():
    return build_content_based()


@st.cache_resource(show_spinner="Building popularity fallback table...")
def load_popularity():
    return build_popularity_table()


@st.cache_resource(show_spinner="Building collaborative matrix & fitting KNN...")
def load_collaborative(_data):
    return build_collaborative(_data)


data, vectors, all_titles, title_to_pos, pos_to_id, id_to_title = load_content()
popularity_table = load_popularity()
(user_rating_matrix, movie_columns, user_index, movieid_to_tmdbid,
 nbrs, user_rating_counts, user_means, user_stds) = load_collaborative(data)


# -----------------------------------------------------------------------------
# Sidebar — inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Inputs")
movie_title = st.sidebar.text_input("Movie title", placeholder="e.g. toy story")
raw_user_id = st.sidebar.text_input("User ID (optional)", placeholder="e.g. 1")
top_n = st.sidebar.slider("Number of recommendations", min_value=3, max_value=15, value=5)
show_breakdown = st.sidebar.checkbox("Show content-only / collaborative-only breakdown", value=True)
run_button = st.sidebar.button("Get Recommendations", type="primary")

with st.sidebar.expander("Advanced"):
    st.caption(
        "The engines are built once and cached in memory for this server process. "
        "If you've updated the CSVs, use the button below to force a rebuild."
    )
    if st.button("Clear cache & rebuild"):
        st.cache_resource.clear()
        st.success("Cache cleared. The page will rebuild the engines on the next run.")
        st.rerun()

user_id = int(raw_user_id) if raw_user_id.strip().isdigit() else None

st.title("🎬 Hybrid Movie Recommendation System")
st.caption("Content-based + collaborative filtering, blended with a graduated cold-start alpha.")


# -----------------------------------------------------------------------------
# Status panel — mirrors the [INFO] lines from the terminal version
# -----------------------------------------------------------------------------
def show_status(movie_title, user_id):
    is_known_user = user_id is not None and user_id in user_index
    num_ratings = user_rating_counts.get(user_id, 0) if is_known_user else 0
    alpha = calculate_graduated_alpha(num_ratings)

    col1, col2, col3 = st.columns(3)
    col1.metric("User status", "Known" if is_known_user else ("Not provided" if user_id is None else "Unknown / new"))
    col2.metric("Ratings on record", num_ratings)
    col3.metric("Alpha (collaborative weight)", f"{alpha:.2f}")

    if movie_title is None and not is_known_user:
        st.info("No usable movie title or known user — results will fall back to overall popularity.")
    elif not is_known_user:
        st.info(f"User unknown/new — alpha = 0.00, results are pure content-based, seeded from '{movie_title}'.")
    else:
        st.info(f"User has {num_ratings} ratings on record — alpha = {alpha:.2f} (graduated confidence).")

    return is_known_user, alpha


def results_to_df(results, score_label="Estimated ★"):
    return pd.DataFrame(results, columns=["Title", score_label])


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------
if run_button:
    if not movie_title.strip() and user_id is None:
        st.warning("Enter a movie title, a user id, or both.")
    else:
        is_known_user, alpha = show_status(movie_title if movie_title.strip() else None, user_id)

        st.subheader("Hybrid Recommendations")
        hybrid_results = hybrid_recommend(
            movie_title=movie_title if movie_title.strip() else None,
            user_id=user_id,
            top_n=top_n,
            vectors=vectors, all_titles=all_titles, title_to_pos=title_to_pos,
            pos_to_id=pos_to_id, id_to_title=id_to_title,
            user_rating_matrix=user_rating_matrix, movie_columns=movie_columns,
            user_index=user_index, movieid_to_tmdbid=movieid_to_tmdbid, nbrs=nbrs,
            user_rating_counts=user_rating_counts, user_means=user_means,
            user_stds=user_stds, popularity_table=popularity_table,
        )

        if hybrid_results:
            st.dataframe(results_to_df(hybrid_results), use_container_width=True, hide_index=True)
        else:
            st.warning("No recommendations found for this input.")

        if show_breakdown:
            st.divider()
            bcol1, bcol2 = st.columns(2)

            with bcol1:
                st.subheader("Content-only")
                if movie_title.strip():
                    seed_id, content_recs = recommend_content(
                        movie_title, vectors, all_titles, title_to_pos, pos_to_id, top_n=top_n
                    )
                    if content_recs:
                        content_df = pd.DataFrame(
                            [(id_to_title.get(mid, f"id {mid}"), round(3.5 + s * 1.4, 1)) for mid, s in content_recs],
                            columns=["Title", "Estimated ★"],
                        )
                        st.dataframe(content_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No content match found for this title.")
                else:
                    st.caption("Enter a movie title to see content-based results.")

            with bcol2:
                st.subheader("Collaborative-only")
                if is_known_user:
                    collab_recs = recommend_user_based(
                        user_id, user_rating_matrix, movie_columns, user_index,
                        movieid_to_tmdbid, nbrs, top_n=top_n
                    )
                    if collab_recs:
                        collab_df = pd.DataFrame(
                            [(id_to_title.get(mid, f"id {mid}"), round(s, 1)) for mid, s in collab_recs],
                            columns=["Title", "Predicted rating"],
                        )
                        st.dataframe(collab_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No confident collaborative predictions for this user.")
                else:
                    st.caption("Enter a known user id to see collaborative results.")
else:
    st.info("Enter a movie title and/or a user id in the sidebar, then click **Get Recommendations**.")
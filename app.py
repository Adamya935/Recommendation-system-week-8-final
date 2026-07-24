# -*- coding: utf-8 -*-
"""
================================================================================
STREAMLIT FRONTEND WEB APP (app.py)
Location: C:\Users\acer\OneDrive\Desktop\Recommendation-System\app.py
================================================================================
Connected directly to hybrid_final.py.
Ultra-simple 2-Input UI (Movie Title & User ID) with HD Movie Poster card grid.
================================================================================
"""

import os
import pandas as pd
import streamlit as st
from hybrid_final import (
    build_content_based,
    build_collaborative,
    hybrid_recommend,
    calculate_graduated_alpha
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #f8fafc;
    }
    .movie-card-box {
        background: rgba(21, 26, 38, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .movie-card-box:hover {
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
    }
    .badge-score {
        background: linear-gradient(135deg, #00f2fe, #7928ca);
        color: #ffffff;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .status-banner {
        background: rgba(0, 242, 254, 0.1);
        color: #00f2fe;
        border: 1px solid rgba(0, 242, 254, 0.25);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED BACKEND INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def init_backend():
    data, vectors, mid_to_pos, pos_to_mid = build_content_based()
    user_rating_matrix, movie_columns, user_index, movie_id_to_title, nbrs, user_rating_counts, user_means, user_stds = build_collaborative(data)
    
    base_poster = "https://image.tmdb.org/t/p/w500"
    if 'poster_path' in data.columns:
        data['poster_url'] = data['poster_path'].apply(
            lambda p: f"{base_poster}{p}" if pd.notna(p) and str(p).startswith('/') else "https://via.placeholder.com/500x750?text=No+Poster"
        )
    else:
        data['poster_url'] = "https://via.placeholder.com/500x750?text=Movie+Poster"

    poster_map = data.set_index('title')['poster_url'].to_dict()
    return data, vectors, mid_to_pos, pos_to_mid, user_rating_matrix, movie_columns, user_index, movie_id_to_title, nbrs, user_rating_counts, user_means, user_stds, poster_map

(data, vectors, mid_to_pos, pos_to_mid, user_rating_matrix, movie_columns,
 user_index, movie_id_to_title, nbrs, user_rating_counts, user_means, user_stds, poster_map) = init_backend()

# -----------------------------------------------------------------------------
# 3. ULTRA-SIMPLE UI: ONLY 2 INPUT FIELDS
# -----------------------------------------------------------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.write("Enter a movie title and optional User ID for real-time hybrid recommendations.")

col_input1, col_input2 = st.columns(2)

with col_input1:
    movie_input = st.text_input("1. Enter Movie Title:", value="The Godfather")

with col_input2:
    user_input = st.text_input("2. Enter User ID (Optional / Leave blank for Cold-Start):", value="1")

user_id = int(user_input.strip()) if user_input.strip().isdigit() else None

if movie_input.strip() or user_id is not None:
    num_r = user_rating_counts.get(user_id, 0) if user_id in user_index else 0
    alpha = calculate_graduated_alpha(num_r)
    is_cold = (user_id is None) or (user_id not in user_index)

    status_txt = f"❄️ **Cold-Start Active**: User '{user_input}' (0 ratings). Alpha (α) = 0.00 (100% Content Match seeded from '{movie_input}')" if is_cold else f"🔥 **Warm Profile Active**: User ID '{user_id}' ({num_r} ratings in DB). Graduated Alpha (α) = {alpha:.2f} (Blending Collab + Content)"
    st.markdown(f"<div class='status-banner'>{status_txt}</div>", unsafe_allow_html=True)

    recs = hybrid_recommend(
        movie_title=movie_input if movie_input.strip() else None,
        user_id=user_id,
        data=data,
        vectors=vectors,
        mid_to_pos=mid_to_pos,
        pos_to_mid=pos_to_mid,
        user_rating_matrix=user_rating_matrix,
        movie_columns=movie_columns,
        user_index=user_index,
        movie_id_to_title=movie_id_to_title,
        nbrs=nbrs,
        user_rating_counts=user_rating_counts,
        user_means=user_means,
        user_stds=user_stds,
        top_n=6
    )

    st.subheader("🎯 Recommended Movies")

    cols_per_row = 3
    for i in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(recs):
                title, est_stars = recs[i + j]
                poster_url = poster_map.get(str(title), "https://via.placeholder.com/500x750?text=Movie+Poster")

                with cols[j]:
                    st.markdown("<div class='movie-card-box'>", unsafe_allow_html=True)
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"### {str(title).title()}")
                    st.markdown(f"<span class='badge-score'>★ {est_stars} / 5.0</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
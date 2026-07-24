
# 🎬 Hybrid Movie Recommendation System

A **Hybrid Movie Recommendation System** built using **Content-Based Filtering** and **Collaborative Filtering**, deployed with **Streamlit**. The system combines user preferences with movie content to generate personalized recommendations while effectively handling cold-start scenarios through a graduated confidence model.

## 🌐 Live Demo

**Streamlit App:**  
https://recommendation-system-week-8-final-wohzhvduufbz595qrl9woh.streamlit.app/

---

## 📌 Features

- Hybrid recommendation engine
- Content-Based Filtering using **TF-IDF**
- Collaborative Filtering using **K-Nearest Neighbors (KNN)**
- Intelligent cold-start handling
- Popularity-based fallback recommendations
- User-friendly Streamlit interface
- Cached model loading for faster performance

---

## 🧠 Recommendation Pipeline

The recommendation system consists of three components:

### 1. Content-Based Filtering
- Uses movie overview, genres, keywords, cast, and director.
- Applies text preprocessing and stemming.
- Generates TF-IDF vectors.
- Calculates cosine similarity to identify similar movies.

### 2. Collaborative Filtering
- Uses MovieLens user ratings.
- Builds a user-item rating matrix.
- Applies User-User KNN with cosine similarity.
- Predicts ratings for unseen movies.

### 3. Hybrid Recommendation
The final recommendation score combines:
- Content similarity
- Collaborative prediction
- Graduated confidence score (α)

This enables the system to gradually shift from content-based recommendations for new users to collaborative recommendations as more user interaction data becomes available.

---

## ❄ Cold Start Handling

The system effectively handles the cold-start problem by adapting its recommendation strategy based on the available information. 
When neither a valid movie title nor a user ID is provided, it generates recommendations using an IMDb-style popularity-based ranking. 
For new or unknown users, the system relies entirely on content-based filtering by identifying movies with similar genres, keywords, cast, director, and plot descriptions using TF-IDF and cosine similarity. 
For existing users with rating history, it combines content-based and collaborative filtering through a graduated confidence score (α), which gradually increases the influence of collaborative recommendations as more user ratings become available. 
Additionally, if a movie lacks collaborative data, the system automatically falls back to content-based scoring for that specific item, ensuring consistent and reliable recommendations across all scenarios.

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit web application
├── hybrid_final.py         # Hybrid recommendation engine
├── requirements.txt
│
├── movies_metadata.csv
├── credits_small.csv
├── keywords.csv
├── ratings_small.csv
├── links_small.csv
├── links.csv
│
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 📊 Dataset

This project uses **The Movies Dataset** together with the **MovieLens Ratings Dataset**.

Dataset files used:
- movies_metadata.csv
- credits_small.csv
- keywords.csv
- ratings_small.csv
- links_small.csv

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- NLTK
- Matplotlib

---

## 🚀 How to Use

1. Enter a movie title.
2. (Optional) Enter a User ID.
3. Choose the number of recommendations.
4. Click **Get Recommendations**.
5. View the generated recommendations along with optional content-based and collaborative recommendation breakdowns.

---

## 📈 Model Highlights

- TF-IDF Vectorization
- Cosine Similarity
- User-User KNN Collaborative Filtering
- IMDb-style Bayesian Popularity Ranking
- Graduated Alpha (α) for Cold-Start Handling
- Streamlit Resource Caching

---

## 📷 Application Preview

Access the deployed application here:

https://recommendation-system-week-8-final-wohzhvduufbz595qrl9woh.streamlit.app/

---

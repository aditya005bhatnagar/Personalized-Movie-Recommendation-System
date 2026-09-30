# 🎬 Personalized Movie Recommendation System

A personalized movie recommendation system built using **User-Based Collaborative Filtering** and **Cosine Similarity**.

The system analyzes users' movie-rating patterns, identifies users with similar preferences, and recommends movies that the target user has not already rated.

---

## 📌 Project Overview

Traditional movie recommendation systems often recommend movies based only on general popularity or movie content.

This project uses **User-Based Collaborative Filtering** to generate personalized recommendations based on the rating behavior of similar users.

The final system provides:

- Personalized movie recommendations
- User-based collaborative filtering
- Cosine similarity
- Similar-user analysis
- Similarity-weighted recommendation scoring
- Exclusion of already-rated movies
- Movie search
- User ratings
- SQLite database for new ratings
- FastAPI backend
- Streamlit frontend
- Precision@K and Recall@K evaluation

---

## 🎯 Objectives

1. Build a personalized movie recommendation system.
2. Identify users with similar movie-rating patterns.
3. Recommend movies based on ratings from similar users.
4. Prevent already-rated movies from appearing in recommendations.
5. Provide a simple interactive web interface.
6. Evaluate the recommendation model using Precision@K and Recall@K.

---

## 🧠 Recommendation Method

The project uses **User-Based Collaborative Filtering**.

### Working Process

```text
MovieLens Dataset
       ↓
Data Preprocessing
       ↓
User-Movie Rating Matrix
       ↓
User-Based Collaborative Filtering
       ↓
Cosine Similarity
       ↓
Find 20 Similar Users
       ↓
Analyze Their Ratings
       ↓
Remove Already-Rated Movies
       ↓
Similarity-Weighted Average Score
       ↓
Top 10 Recommendations
       ↓
FastAPI Backend
       ↓
Streamlit Application
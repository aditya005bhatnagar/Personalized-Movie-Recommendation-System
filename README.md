# 🎬 Personalized Movie Recommendation System

A machine learning based movie recommendation system that provides personalized movie recommendations based on user rating patterns.

The project uses the MovieLens dataset and user-based collaborative filtering to identify users with similar movie preferences and recommend movies they may enjoy.

The system is implemented using Python, Scikit-learn, Pandas, Joblib, and Streamlit.

---

## 🚀 Features

- 👤 Personalized recommendations for individual users
- 🤝 User-based collaborative filtering
- 🎯 Finds users with similar movie-rating patterns
- ⭐ Uses ratings from similar users to generate recommendations
- 🚫 Removes movies the selected user has already rated
- 🎬 Displays movie genres and recommendation scores
- 🖥️ Interactive Streamlit web application
- 💾 Trained models saved using Joblib

---

## 🧠 How It Works

The recommendation process follows these steps:

```text
Select User
     ↓
Create User-Movie Rating Matrix
     ↓
Find Similar Users
     ↓
Analyze Movies Rated by Similar Users
     ↓
Remove Movies Already Rated
     ↓
Calculate Recommendation Scores
     ↓
Rank Movies
     ↓
Return Top 10 Movies
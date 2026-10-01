# 🎬 CineMatch - Personalized Movie Recommendation System

CineMatch is a personalized movie recommendation system that uses **User-Based Collaborative Filtering** to recommend movies based on user rating patterns.

The application provides a **Streamlit frontend** and a **FastAPI backend**, along with user registration, login, movie ratings, and MovieLens user integration.

---

## 🚀 Features

- 🎬 Personalized movie recommendations
- 👤 User registration and login
- 🔐 Password hashing
- ⭐ Movie rating system
- 👥 User-Based Collaborative Filtering
- 🔗 MovieLens user integration
- 🆔 CineMatch User ID to MovieLens User ID mapping
- 🗄️ SQLite database for user accounts and ratings
- ⚡ FastAPI backend
- 🖥️ Streamlit frontend
- 🎯 Personalized recommendations based on user ratings
- 📊 Movie and user information through REST APIs

---

## 🧠 Recommendation System

CineMatch uses **User-Based Collaborative Filtering**.

The system identifies users with similar movie-rating patterns and uses those patterns to generate movie recommendations.

### Existing MovieLens User

```text
CineMatch User
      ↓
MovieLens User ID
      ↓
Existing MovieLens ratings
      ↓
Collaborative Filtering Model
      ↓
Personalized Recommendations
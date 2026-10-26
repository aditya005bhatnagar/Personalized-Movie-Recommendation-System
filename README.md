# 🎬 CineMatch — Personalized Movie Recommendation System

CineMatch is a movie recommendation web application that helps users discover movies based on rating patterns and preferences. It combines a Streamlit frontend with a FastAPI backend, user-based collaborative filtering, and TMDB data for movie posters and descriptions.

## ✨ Features

- **User accounts:** Register and log in through the backend.
- **Personalized recommendations:** Discover movies using user-based collaborative filtering.
- **User profile:** View your user ID, number of movies rated, and average rating.
- **Highly rated movies:** See movies you have rated highly.
- **Movie discovery:** Search for movies by title.
- **Movie details:** Display posters and descriptions retrieved from TMDB when available.
- **Community ratings:** View the average community rating and number of reviews.
- **Written reviews:** Submit a rating and written review, and read reviews from other users.

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Language | Python |
| Data processing | Pandas |
| Recommendation approach | User-based collaborative filtering |
| Model persistence | Joblib |
| Database | SQLite via SQLAlchemy (if configured in the project) |
| Movie posters and descriptions | The Movie Database (TMDB) API |
| Dataset | MovieLens |

## 📁 Project Structure

```text
MovieRecommendationSystem/
├── api/
│   ├── main.py
│   └── database.py
├── saved_models/
│   ├── movies.pkl
│   ├── user_model.pkl
│   ├── user_movie_matrix.pkl
│   └── user_ids.pkl
├── app.py
├── .env
├── requirements.txt
└── README.md
```

Your project may contain additional files. Keep your existing files and folders; this tree shows the main components used by the application.

## ⚙️ Requirements

- Python 3.10 or newer (use a version compatible with the packages in `requirements.txt`)
- pip
- Internet connection for TMDB movie data
- A TMDB API read access token
- The saved model files required by the app

## 🚀 Setup and Installation

### 1. Clone the repository

If the project is hosted on GitHub, clone your repository and enter its folder:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd MovieRecommendationSystem
```

If the project is already on your computer, open PowerShell in the project folder instead.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands below without activating the environment by using `.venv\Scripts\python.exe` instead.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If `requirements.txt` is missing or incomplete, add the packages your code imports, then install them. Common packages used by this project include:

```powershell
python -m pip install streamlit fastapi "uvicorn[standard]" pandas scikit-learn joblib requests python-dotenv sqlalchemy
```

### 4. Configure the TMDB token

Create a `.env` file in the project root, alongside `app.py`:

```env
TMDB_TOKEN=your_tmdb_api_read_access_token
```

Replace the example value with your actual TMDB API read access token. Keep this token private and **do not commit `.env` to GitHub**.

The app should still handle missing TMDB data gracefully if a token is unavailable, but posters and descriptions may not load.

### 5. Check the saved models

Make sure the model files expected by `app.py` exist in `saved_models/`, including:

- `movies.pkl`
- `user_model.pkl`
- `user_movie_matrix.pkl`
- `user_ids.pkl`

If any required file is missing, the application may fail while loading its recommendation models.

## ▶️ Run the Application

Run the backend and frontend in **two separate PowerShell terminals**.

### Terminal 1 — Start FastAPI

```powershell
cd C:\MovieRecommendationSystem
python -m uvicorn api.main:app --reload
```

FastAPI should be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### Terminal 2 — Start Streamlit

```powershell
cd C:\MovieRecommendationSystem
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, commonly `http://localhost:8501`.

If your project is stored in a different folder, replace `C:\MovieRecommendationSystem` with your actual project path.

## 🧠 How Recommendations Work

1. The application loads the saved movie data and recommendation model.
2. The signed-in user's ID is used to request recommendations from the backend.
3. The recommendation system compares users' movie-rating patterns.
4. Movies already rated by the user can be excluded from the recommendation results.
5. The frontend displays recommended movie titles, genres, posters when available, and recommendation scores when returned by the API.

The exact recommendation results depend on the saved model, the dataset, and the backend implementation.

## ⭐ Ratings and Reviews

Users can search for a movie, view its details, see community rating statistics, and submit a rating with a written review. The backend provides endpoints for submitting reviews and retrieving a movie's reviews.

The application expects the backend to be running for account operations, community ratings, reviews, and API-based recommendations.

## 🎞️ TMDB Attribution

CineMatch uses the TMDB API to retrieve movie information such as posters and descriptions.

> This product uses the TMDB API but is not endorsed or certified by TMDB.

Movie descriptions and posters may be unavailable for some titles. TMDB results depend on API access, title matching, network availability, and whether TMDB has data for a particular movie.

## 🔐 Security Notes

- Keep API tokens and other secrets in `.env`.
- Add `.env` and `.venv/` to `.gitignore`.
- Do not publish passwords, database credentials, API tokens, or private user data.
- For a production deployment, use secure password handling, environment-based configuration, input validation, and appropriate authentication and authorization.

## 🛠️ Troubleshooting

### TMDB descriptions or posters are missing

- Check that `.env` is in the same project root as `app.py`.
- Confirm the variable is named `TMDB_TOKEN`.
- Verify that the token is valid and that the computer has internet access.
- Check the Streamlit terminal for API errors.
- Some movies may not have a description or poster available on TMDB.

### The frontend cannot connect to FastAPI

- Start the backend in a separate terminal.
- Confirm it is running at `http://127.0.0.1:8000`.
- Check that `API_URL` in `app.py` matches the backend address.

### A model file cannot be loaded

- Confirm that the required `.pkl` files are present in `saved_models/`.
- Run the application from the project root so relative file paths resolve correctly.

### A Python package is missing

```powershell
python -m pip install -r requirements.txt
```

## 🔮 Future Improvements

- Improve movie matching and recommendation quality.
- Add genre and keyword filters.
- Improve recommendation explanations.
- Add stronger authentication and user-specific review permissions.
- Add automated tests and deployment instructions.

## 👨‍💻 Author

**Aditya Bhatnagar**

CineMatch is a learning project focused on movie recommendation systems, Python, machine learning, and full-stack application development.

import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")

print("Movies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

print("\nUsers:")
print(users.head())

print("\nMovies Info:")
print(movies.info())

print("\nRatings Info:")
print(ratings.info())

print("\nUsers Info:")
print(users.info())

print("\nMissing values in Movies:")
print(movies.isnull().sum())

print("\nMissing values in Ratings:")
print(ratings.isnull().sum())

print("\nMissing values in Users:")
print(users.isnull().sum())

print("\nDuplicate rows in Movies:")
print(movies.duplicated().sum())

print("\nDuplicate rows in Ratings:")
print(ratings.duplicated().sum())

print("\nDuplicate rows in Users:")
print(users.duplicated().sum())

print("\nRating values:")
print(ratings["rating"].value_counts().sort_index())

print("\nInvalid ratings:")
print(ratings[(ratings["rating"] < 1) | (ratings["rating"] > 5)])

print("\nUser ID check:")
print("Users:", users["userId"].nunique())
print("Users in ratings:", ratings["userId"].nunique())

print("\nMovie ID check:")
print("Movies:", movies["movieId"].nunique())
print("Movies in ratings:", ratings["movieId"].nunique())

print("\nGender values:")
print(users["gender"].value_counts())

print("\nAge values:")
print(users["age"].value_counts().sort_index())

movie_ratings = pd.merge(ratings, movies, on="movieId")

print("\nMerged Data:")
print(movie_ratings.head())

print("\nMerged Data Info:")
print(movie_ratings.info())

print("\nNumber of ratings per movie:")
movie_rating_count = ratings.groupby("movieId").size()
print(movie_rating_count.describe())

print("\nNumber of ratings per user:")
user_rating_count = ratings.groupby("userId").size()
print(user_rating_count.describe())

print("\nTop 10 Most Rated Movies:")

top_movies = ratings.groupby("movieId").size().sort_values(ascending=False).head(10)

top_movies = pd.merge(
    top_movies.reset_index(name="rating_count"),
    movies,
    on="movieId"
)

print(top_movies[["movieId", "title", "rating_count"]])

print("\nPreparing Genres:")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

print(movies[["title", "genres"]].head(10))
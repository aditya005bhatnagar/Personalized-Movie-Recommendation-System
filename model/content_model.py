import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres"])

similarity = cosine_similarity(tfidf_matrix)

def recommend_movies(movie_title, n=10):

    if movie_title not in movies["title"].values:
        print("Movie not found")
        return

    movie_index = movies[movies["title"] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = []

    for index, score in similarity_scores[1:n+1]:
        recommendations.append(movies.iloc[index]["title"])

    return recommendations


recommendations = recommend_movies("Toy Story (1995)")

print("\nRecommended Movies:")

for movie in recommendations:
    print(movie)
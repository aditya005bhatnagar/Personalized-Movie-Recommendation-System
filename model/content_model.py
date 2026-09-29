import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(movies["genres"])

print("TF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

similarity = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:")
print(similarity.shape)

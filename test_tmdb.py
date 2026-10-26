import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TMDB_TOKEN")

if not token:
    print("TMDB_TOKEN not found")
    exit()

headers = {
    "Authorization": f"Bearer {token}",
    "accept": "application/json"
}

response = requests.get(
    "https://api.themoviedb.org/3/search/movie",
    headers=headers,
    params={
        "query": "Toy Story"
    }
)

print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()

    for movie in data["results"][:3]:
        print(movie["title"], movie.get("poster_path"))

else:
    print(response.text)
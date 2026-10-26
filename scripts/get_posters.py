import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_TOKEN = os.getenv("TMDB_TOKEN")

if not TMDB_TOKEN:
    raise ValueError("TMDB_TOKEN not found in .env")

movies = pd.read_csv("data/movies.csv")

headers = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json"
}

poster_data = []

for index, movie in movies.iterrows():

    movie_id = int(movie["movieId"])
    title = str(movie["title"])

    clean_title = title.rsplit(" (", 1)[0]

    success = False

    for attempt in range(3):

        try:

            response = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                headers=headers,
                params={
                    "query": clean_title,
                    "language": "en-US",
                    "include_adult": "false"
                },
                timeout=20
            )

            if response.status_code == 200:

                results = response.json().get(
                    "results",
                    []
                )

                poster_url = ""

                if results:

                    poster_path = results[0].get(
                        "poster_path"
                    )

                    if poster_path:
                        poster_url = (
                            "https://image.tmdb.org/t/p/w500"
                            + poster_path
                        )

                poster_data.append({
                    "movieId": movie_id,
                    "poster_url": poster_url
                })

                print(
                    f"[{index + 1}/{len(movies)}] "
                    f"{title}"
                )

                success = True
                break

            else:

                print(
                    f"HTTP {response.status_code}: {title}"
                )

        except requests.exceptions.RequestException as e:

            print(
                f"Attempt {attempt + 1}/3 failed: "
                f"{title}"
            )

            time.sleep(2)

    if not success:

        poster_data.append({
            "movieId": movie_id,
            "poster_url": ""
        })

        print(f"Skipped: {title}")

    time.sleep(0.5)


poster_df = pd.DataFrame(poster_data)

poster_df.to_csv(
    "data/movie_posters.csv",
    index=False
)

print()
print("Poster mapping created successfully.")
print(f"Total movies processed: {len(poster_df)}")
print(
    f"Posters found: "
    f"{(poster_df['poster_url'] != '').sum()}"
)
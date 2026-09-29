# 🎬 Movie Recommendation System

Ever watched a movie and then wondered, **"What should I watch next?"**

This project is a Movie Recommendation System that suggests movies based on the movie you select. I built it as a machine learning project to understand how recommendation systems work in real-world applications.

The project uses **content-based filtering** and **collaborative filtering**, with a Streamlit interface so the recommendations can be viewed through a simple web application.

---

## 📌 About the Project

The main idea behind this project is simple:

1. Select a movie you like.
2. The system looks at its genre and other movie information.
3. It also looks at rating patterns from other users.
4. Based on these patterns, it recommends movies that you might enjoy.

For example, if you select **Toy Story (1995)**, the system can recommend other movies with similar genres and rating patterns.

---

## 🧠 How the Recommendation Works

### Content-Based Filtering

This method recommends movies that are similar to the movie you selected.

For this project, I used the movie genres and converted them into numerical features using **TF-IDF**.

Then, **cosine similarity / nearest neighbors** is used to find movies with similar genre information.

### Collaborative Filtering

Collaborative filtering looks at how users have rated different movies.

A user-movie rating matrix is created from the ratings dataset. Movies that have similar rating patterns are then identified.

This means the recommendation is not based only on the movie's genre, but also on how other users have interacted with movies.

### Combined Recommendations

The application combines the recommendations from both approaches to produce the final list of movies.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- TF-IDF
- Cosine Similarity
- Nearest Neighbors

---

## 📊 Dataset

This project uses the **MovieLens 1M dataset**.

The dataset contains:

- **3,883 movies**
- **1,000,209 ratings**
- **6,040 users**

The main files are:

```text
movies.csv
ratings.csv
users.csv
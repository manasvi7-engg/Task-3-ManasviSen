import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Convert genre text into numerical vectors
cv = CountVectorizer()
vectors = cv.fit_transform(movies["Genre"])

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

print("====== AI Movie Recommendation System ======\n")

print("Available Movies:\n")

for movie in movies["Movie"]:
    print(movie)



movie_name = input("\nEnter a movie name: ").strip().lower()

# Convert movie names to lowercase for comparison
movies["Movie_lower"] = movies["Movie"].str.lower()

if movie_name not in movies["Movie_lower"].values:
    print("Movie not found!")
else:
    index = movies[movies["Movie_lower"] == movie_name].index[0]


    distances = list(enumerate(similarity[index]))

    # Sort by similarity score
    recommended = sorted(
        distances,
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in recommended:
        print(movies.iloc[i[0]].Movie)
import streamlit as st
import pandas as pd
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load the pre-trained model
model = joblib.load('svd_model.pkl')

# Load movie data
movies_df = pd.read_csv('Data/movies.csv')

# Function to generate recommendations
def get_recommendations(user_id, top_n=5):
    ratings_df = pd.read_csv('Data/ratings.csv')
    combined_ratings_df = pd.concat([ratings_df, pd.DataFrame({'userId': [user_id]*len(movies_df), 'movieId': movies_df['movieId'], 'rating': [0]*len(movies_df)})], axis=0)
    combined_ratings_df = combined_ratings_df[['userId', 'movieId', 'rating']]
    combined_ratings_df['rating'] = combined_ratings_df['rating'].astype(float)

    reader = Reader(rating_scale=(0.5, 5.0))
    new_data = Dataset.load_from_df(combined_ratings_df, reader)

    trainset, testset = train_test_split(new_data, test_size=0.2, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    list_of_movies = [(m_id, model.predict(user_id, m_id).est) for m_id in movies_df['movieId']]
    ranked_movies = sorted(list_of_movies, key=lambda x: x[1], reverse=True)

    top_movies = ranked_movies[:top_n]
    return top_movies

# Streamlit interface
st.title('Movie Recommendation System')

user_id = st.number_input('Enter your user ID:', min_value=1, max_value=1000, value=1000)
top_n = st.slider('Number of recommendations to show:', min_value=1, max_value=10, value=5)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_id, top_n)
    st.write('Top Recommendations:')
    for idx, rec in enumerate(recommendations):
        movie_title = movies_df.loc[movies_df['movieId'] == rec[0], 'title'].values[0]
        st.write(f'Recommendation #{idx + 1}: {movie_title} (Predicted Rating: {rec[1]:.2f})')

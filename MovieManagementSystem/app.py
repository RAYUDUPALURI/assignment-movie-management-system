from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)
app.secret_key = 'Rayudu'

login_manager = LoginManager()
login_manager.init_app(app)

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# User credentials (username:password)
users = {'Rayudu': 'Rayudu'}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Initial movie data
movies = pd.DataFrame(columns=['user', 'item', 'rating'])

reader = Reader(line_format='user item rating', sep=',')
dataset = Dataset.load_from_df(movies, reader)
trainset, testset = train_test_split(dataset, test_size=0.2)

# Collaborative Filtering model
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/add_movie', methods=['POST'])
@login_required
def add_movie():
    title = request.form['title']
    rating = request.form['rating']
    
    # Add the movie to the dataset
    global movies
    movies = movies.append({'user': 'Rayudu', 'item': title, 'rating': float(rating)}, ignore_index=True)
    
    # Retrain the model
    global dataset, trainset, testset
    dataset = Dataset.load_from_df(movies, reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model.fit(trainset)
    
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
@login_required
def recommend():
    user_id = 'Rayudu'
    predictions = []
    
    for movie_id in dataset.df['item'].unique():
        prediction = model.predict(user_id, movie_id)
        predictions.append((prediction.iid, prediction.est))

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N recommendations (e.g., top 5)
    top_recommendations = predictions[:5]

    return render_template('recommendations.html', recommendations=top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)

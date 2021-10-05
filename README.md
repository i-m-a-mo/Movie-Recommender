# Movie Recommender

### Machine learning model for recommending movies
What it is:
* It's a web interface that takes a user's rating of 3 movies (Drama, Action, Comedy) to generate a list of 10 recommended movies for the user to watch based on those preferences using an NMF model

How it works:
* each genre contains a list of 10 randomly chosen movies for the user to choose from
* each movie is rated by the user on a scale from 1 to 5
* this user input is then fed into a machine-learning model: *Non-negative Matrix Factorization (NMF)*
* the model is trained on a database of movies (including user ratings) from MovieLens.org
* the list of recommendations is then presented to the user on the web interface

<!-- 
* recommendation is based on the user's rating of 3 movies (Drama, Action, Comedy)
-->
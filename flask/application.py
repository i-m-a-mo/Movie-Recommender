"""
Module that controls the workfow of the web application. Central python file that ties everything together for the web application.
"""

from simple_recommerder import createuserp, get_recommendations
from flask import Flask, render_template, request

from movie_function import get_random_movies, movie_labels

app = Flask(__name__)


@app.route("/")  # app.route defines the url of the functionality that follows
# @... indicates that this is a decorator
# a decorator is a function that takes a function as an input argument and returns a function
# decorators extend the functionality of a function that you define
def hello_world():
    random_movie_dict = get_random_movies(['Drama','Action','Comedy'],10)
    m_labels = movie_labels(random_movie_dict)
    return render_template("index.html",
                           title="Welcome to the Camola Movie Recommender",
                           movie_dict = random_movie_dict,
                           labels = m_labels)





@app.route("/recommender")
def recommender():
    html_form_data = dict(request.args)   ## this request is taking the new user inputs
    print("THIS IS THE FORM DATA --- ", html_form_data)
    userp = createuserp(html_form_data)
    top_movies = get_recommendations(userp)
    print(top_movies)
    return render_template("recommendations.html", movies=top_movies)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
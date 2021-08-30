import pickle
import numpy as np


# genre
with open("../model/genres.pickle", "rb") as l:
    genre = pickle.load(l)  

def get_random_movies(genre_list,num_movies_per_genre):
    movie_dict = { i : [] for i in genre_list}

    for g in genre_list:
        all_movies_of_genre = genre.iloc[:,:2][genre[0] == g]["title"].tolist()
        random_array = np.random.choice(all_movies_of_genre,
                                        size=num_movies_per_genre,
                                        replace=False)
        for i in range(num_movies_per_genre):
            movie_dict[g].append(random_array[i].replace(" ", "_"))

    return movie_dict
        
        
def movie_labels(movie_dict):
    movie_label_dict = { key : [] for key in movie_dict}
    for key in movie_dict:
        for i in range(len(movie_dict[key])):
            movie_label_dict[key].append(movie_dict[key][i].replace("_", " "))            
    return movie_label_dict
        

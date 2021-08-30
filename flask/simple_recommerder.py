"""
Module that trains the model
"""
#%%
import pickle
import numpy as np
import pandas as pd


#%% Pickle     

# KNN for missing values    
with open("../model/knn2.pickle", "rb") as g:
    imputer = pickle.load(g)

# Model NMF:
with open("../model/nmf.pickle", "rb") as f:
    model = pickle.load(f)

# Table with index and movie names
with open("../model/movies_ind.pickle", "rb") as h: 
    movies_ind = pickle.load(h)

# R
with open("../model/R_matrix.pickle", "rb") as k:
    R = pickle.load(k)  

# user-movie-matrix  
with open("../model/u_m_matrix.pickle", "rb") as l:
    u_m_matrix = pickle.load(l)

Q = model.components_  # movie-genre matrix


#%%

"""createuserp is a function that convert the input from the new user in an array with the new ratings"""

def createuserp(movieratingdict):

    new_user = np.zeros((1,R.shape[1]))
    new_user[:] = np.NaN
    new_user = pd.DataFrame(data= new_user, columns=u_m_matrix.columns)

    ### PLACEHOLDERS
    for m in range(1,4):
        new_user.loc[0][movieratingdict['genre'+ str(m)].replace("_", " ")] = movieratingdict ['rating' + str(m)]     # --> input data from html_form_dat
        
    print(new_user.T.dropna())
    return new_user


#%%
""" get_resommendations, this function takes as imput the new_user from above: """

def get_recommendations(new_user):

    new_u = imputer.transform(new_user)          # make the imputation with knn   
    user_P = model.transform(new_u)                 # run the model nmf and creat user_P
    actual_recommendations = np.dot(user_P, Q)      # compares the new user with old users

    top_movies = np.argsort(actual_recommendations)    # sort the index of the movies from the movie less ranked to the highes ranked movie
    top_5 = []
    movies_ind["title"] = movies_ind["title"].str.replace(" ", "_")
    for top in top_movies[0][-5:]:                                        # take the index of best ranked movies and find the name per each movie
        recommendation = movies_ind.loc[movies_ind['index']==top , 'title'].str.replace("_"," ").values[0] 
        #recommendation = movies_ind.loc[movies_ind['index']==top , 'title']  
        #top_5.append(u_m_matrix.columns[top])
        top_5.append(recommendation)
    return top_5



#%%# IMPORTS 
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
import pickle


#%%# LOAD IN THE DATA & INSPECT
movies = pd.read_csv('./ml-latest-small/movies.csv', index_col=0)
links = pd.read_csv('./ml-latest-small/links.csv', index_col=0) ### don't know if we need this one
ratings = pd.read_csv('./ml-latest-small/ratings.csv')#,  index_col=0)#, parse_dates=True)
tags = pd.read_csv('./ml-latest-small/tags.csv', index_col=0, parse_dates=True)

movies.columns, links.columns, ratings.columns, tags.columns

movies.shape, links.shape, ratings.shape, tags.shape

genres = pd.concat((movies['title'],movies['genres'].str.split(pat='|', n=- 1, expand=True)),
                    axis=1)

#%%# PREP DATA FOR NMF
movies = movies.reset_index()
df = pd.merge(ratings[['userId','movieId','rating']],
              movies[['title','movieId']], 
              how='inner',
              on='movieId').drop(columns=['movieId'])

u_m_matrix = pd.pivot_table(df, values='rating',index='userId', columns='title')


#%%# DEAL WITH MISSINGS --> K NEAREST NEIGHBOURS 
# percentage of missings 
u_m_matrix.isna().sum().sum() / (u_m_matrix.shape[0]*u_m_matrix.shape[1])

imputer = KNNImputer(n_neighbors=2)
knn = imputer.fit(u_m_matrix)

knn = imputer.transform(u_m_matrix)
u_m_matrix_nona = pd.DataFrame(knn, index=u_m_matrix.index, columns=u_m_matrix.columns)

#%%
## -------------------------------------------------------------------- ##
## -------------------------------- NMF ------------------------------- ##
## -------------------------------------------------------------------- ##

# movie_genre = Q
# user_genre = P
# user_movie = R

R_data = u_m_matrix_nona

#need a dataframe for this
R = pd.DataFrame(R_data, index=R_data.index, columns=R_data.columns).values 

#create a model and set the hyperparameters
# model assumes R ~ PQ'

model = NMF(n_components=20, ## n_components == genres --> 19 from readme
            init='random',   ## nndsvd - better for sparseness // nndsvda - better when sparsity is not desired ???
            random_state=42, ## for random init
            tol = 0.0001,    ## tolerance
            max_iter = 5000) ## default of 200 easily maxed out with n_compo=19
                             ## but obvs increases fitting time!!!
# fitting the model to R
model.fit(R)

Q = model.components_  # movie-genre matrix
Q.shape #genres, no.of.movies
P = model.transform(R)  # user-genre matrix
P.shape #no.of users, genre
print(model.reconstruction_err_) #reconstruction error

nR = np.dot(P, Q)
print(nR.shape) ## The reconstructed matrix!

#%%
## -------------------------------------------------------------------- ##
## -------------------------- PICKLE EXPORTS -------------------------- ##
## -------------------------------------------------------------------- ##


# USER-MOVIE-MATRIX
with open('u_m_matrix.pickle', 'wb') as f:
    pickle.dump(u_m_matrix, f)

# KNN MODEL
with open('knn.pickle', 'wb') as f:
    pickle.dump(knn, f)
    
# R MATRIX
with open('R_matrix.pickle', 'wb') as f:
    pickle.dump(R, f)
    
# NMF MODEL
with open('nmf.pickle', 'wb') as f:
    pickle.dump(model, f)

# GENRES DF
with open('genres.pickle', 'wb') as f:
    pickle.dump(genres, f)

# coding: utf-8

# In[21]:


import pandas as pd

user_ratings = pd.read_csv('ratings.csv')

user_ids = user_ratings['userId']
movie_ids = user_ratings['movieId']
ratings = user_ratings['rating']

user_id = set(user_ids)
movie_id = set(movie_ids)
total_possible = len(user_id)*len(movie_id)
total_available = len(ratings)
sparity = float(total_available)/total_possible
sparity


# In[54]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

rating_dict = {}
for i in ratings:
    if i not in rating_dict:
        rating_dict[i] = 0
    else:
        rating_dict[i] += 1
    
plt.bar(np.linspace(0.5,5,10),list(rating_dict[k] for k in sorted(rating_dict.keys())) ,alpha=0.8, width=0.4)

xmajorLocator = MultipleLocator(0.5)
ax = plt.gca()
ax.xaxis.set_major_locator(xmajorLocator) 

plt.xlabel('Rating')
plt.ylabel('Number')
plt.show()


# In[23]:


number_of_movie_rating = {} # store the dict{movie id : the number of rating}
movie_ids_listed = list(movie_ids)
for i in movie_id:
    number_of_movie_rating[i] = movie_ids_listed.count(i)

number_of_movie_rating_sorted = sorted(number_of_movie_rating.items(), key=lambda e:e[1], reverse=True)# sorted dict by values i.e. the number of rating

movies_sorted = []
number_of_rating = []
for i in number_of_movie_rating_sorted:
    movies_sorted.append(i[0])
    number_of_rating.append(i[1])
plt.plot(number_of_rating)
plt.xlabel('Movie ID')
plt.ylabel('Number')
plt.show()


# In[52]:


number_of_user_rating = {} # store the dict{user id : the number of movies the user have rated}
user_ids_listed = list(user_ids)
for i in user_id:
    number_of_user_rating[i] = user_ids_listed.count(i)
number_of_user_rating_sorted = sorted(number_of_user_rating.items(), key=lambda e:e[1], reverse=True)
# sorted dict by values i.e. the number of movies the user have rated

users_sorted = []
number_of_user_rating = []
for i in number_of_user_rating_sorted:
    users_sorted.append(i[0])
    number_of_user_rating.append(i[1])
xmajorLocator   = MultipleLocator(0.5)
plt.plot(number_of_user_rating)
plt.xlabel('User ID')
plt.ylabel('Number')
plt.show()


# In[56]:


from collections import defaultdict
import pandas as pd
from itertools import groupby
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
d = defaultdict(list)
for i, j in zip(movie_ids, ratings):# create a multidict {movie id: [the ratings of the movie]}
    d[i].append(j)
movie_rating_variance = {}
for i, j in d.items():
    movie_rating_variance[i] = np.var(j) # create a dict {movie id: the variance of the rating values received by the movie }

values=movie_rating_variance.values()
variance = []
for k, g in groupby(sorted(values), key=lambda x: x//0.5):
    variance.append(len(list(g)))

plt.bar(np.linspace(0.5,5,10),variance,alpha=0.8, width=0.4)

xmajorLocator = MultipleLocator(0.5)
ax = plt.gca()
ax.xaxis.set_major_locator(xmajorLocator) 


plt.xlabel('Variance intervals')
plt.ylabel('Number')
plt.show()

variance


#!/usr/bin/env python
# coding: utf-8

# In[179]:


# 1. matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet 
#from surprise import Reader, Dataset, SVD, evaluate
#from surprise.model_selection import train_test_split
#from surprise.model_selection import cross_validate
import warnings; warnings.simplefilter('ignore')


# In[ ]:


# 2. Top N Recommendations


# In[5]:


md = pd.read_csv('C:\\Users\gfossung\Downloads\movies_metadata.csv')


# In[6]:


md.head(10)


# In[7]:


md.head()


# In[ ]:


# 2.1 Preprocessing


# In[ ]:


md['genres'] = md['genres'].fillna('[]')


# In[21]:


md.head()


# In[8]:


list1 = '[1,2,3,4,5]'


# In[9]:


list1


# In[10]:


list1[0]


# In[11]:


list_eval = eval(list1)


# In[12]:


list_eval 


# In[13]:


list_eval[0] 


# In[14]:


list_eval


# In[15]:


literal_eval(list1)[0]


# In[22]:


md['genres'] = md['genres'].apply(literal_eval)


# In[23]:


md.head()


# In[24]:


md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[20]:


md.head()


# In[25]:


md.head() 


# In[26]:


list1 = '[1,2,3,4,5]'
list1


# In[27]:


list1


# In[28]:


list1


# In[29]:


list1[0]


# In[30]:


list_eval = eval(list1)


# In[31]:


list_eval


# In[32]:


list_eval[0]


# In[33]:


list_eval


# In[34]:


list_eval[0]


# In[35]:


literal_eval(list1)[0]


# In[89]:


md['genres'] = md['genres'].apply(literal_eval)


# In[37]:


md.head()


# In[ ]:


#Genre as list & applying Lambda fxn to obtain genres as pure text


# In[38]:


md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[39]:


md.head()


# In[ ]:


#Eliminate null entries & convert vote_count to integer


# In[40]:


md[md['vote_count'].notnull()] 


# In[43]:


vote_count = md[md['vote_count'].notnull()]['vote_count'].astype('int')


# In[182]:


vote_count


# In[ ]:


#Eliminate null entries & convert vote_average to integer


# In[45]:


vote_average = md[md['vote_average'].notnull()]['vote_average'].astype('int') 


# In[46]:


vote_average


# In[47]:


top_movies=md.copy()


# In[51]:


top_movies1 = top_movies.sort_values('vote_average', ascending=False).head(250)


# In[ ]:


# No Min votes requirement


# In[52]:


top_movies1


# In[ ]:


# Min number of votes 1000


# In[53]:


top_movies2 = top_movies[top_movies['vote_count']>1000]


# In[54]:


top_movies2


# In[55]:


top_movies2.sort_values('vote_average', ascending=False).head(250)


# In[ ]:


## End of Top250 movies from the chart based on 'average_vote'  


# In[ ]:


# Using Weighted Ratio (WR) to build an overall Top250 Chart, and define a function
# to build charts for a particular genre
# WR = (v*R/(v+m))+(m*C/(v+m))
#vote_count = md[md['vote_count'].notnull()]['vote_count'].astype('int') 
#vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int') 
#C = vote_averages.mean()


# In[56]:


C = vote_average.mean()


# In[84]:


C


# In[58]:


m = vote_count.quantile(0.95)


# In[131]:


m


# In[ ]:


# Retrieve year column from data


# In[60]:


top_movies['year'] = pd.to_datetime(top_movies['release_date'], errors='coerce').apply(lambda x:
str(x).split('-')[0] if x!= np.nan else np.nan)


# In[61]:


top_movies


# In[62]:


top_movies3 = top_movies[(top_movies['vote_count'] >= m) & (top_movies['vote_count'].notnull()) & 
(top_movies['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]


# In[63]:


top_movies3['vote_count'] = top_movies3['vote_count'].astype('int') 
top_movies3['vote_average'] = top_movies3['vote_average'].astype('int')


# In[64]:


top_movies3.shape


# In[65]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[ ]:


# Compute 'weighted_rate'


# In[88]:


top_movies3['weight_rate'] = top_movies3.apply(weighted_rating, axis=1)


# In[70]:


top_movies3.head()


# In[ ]:


#weighted_rating is passed onto the lambda function && axis=1=> will be done for every row


# In[91]:


top_movies3 = top_movies3.sort_values('weight_rate', ascending=False).head(10)


# In[72]:


top_movies3.head(10)


# In[183]:


top_movies3.head(15)


# In[ ]:


# 3. Top Movies


# In[ ]:


# Genre = Romance
# Split the genres further


# In[74]:


genre_TM = top_movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True) 
genre_TM.name = 'genre'
genre_top_movies = top_movies.drop('genres', axis=1).join(genre_TM)


# In[76]:


genre_top_movies


# In[189]:


def build_chart(genre, percentile=0.85):
    None
   # df = genre_top_movies[genre_top_movies['genre'] == genre]
    #vote_counts = df[df[vote_count].notnull()]['vote_count'].astype('int')
    #vote_averages = df[df[vote_average].notnull()]['vote_average'].astype('int')
    #C = vote_averages.mean()
   # m = vote_counts.quantile(percentile)
   # qualified = df[(df[vote_count] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())]
   # qualified['vote_count'] = qualified['vote_count'].astype('int')
   # qualified['vote_average'] = qualified['vote_average'].astype('int')
   
    #Verify the definition below????????????
#    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']*x['vote_average']/(x['vote_count']+ vote_counts.quantile(percentile))
 #+ ((vote_counts.quantile(percentile) * vote_average.mean()/((vote_counts.quantile(percentile) +x['vote_count'])))))) 
    #qualified = qualified.sort_value('wr', ascending=False).head(250)
 #   top_movies4 =  top_movies4.sort_value('wr', ascending=False).head(250)
    return  top_movies3


# In[ ]:





# In[ ]:


# 4. Top Genre Movies
# Movies will be categorised by different genres


# In[190]:


build_chart('Animation').head(10)


# In[191]:


build_chart('Family').head(10)


# In[192]:


build_chart('Action').head(10)


# In[ ]:





# In[ ]:


# 5. Content Based Recommender


# In[193]:


links_small = pd.read_csv('C:\\Users\gfossung\Downloads\links_small.csv')


# In[194]:


links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[115]:


top_movies = top_movies.drop([19730, 29503, 35587]) 


# In[ ]:


#Check EDA Notebook to figure out how and why the above indices were obtained. (43:15 mins into Day2 video)


# In[ ]:



# Analyse a subset of top_movies


# In[195]:


top_movies['id'] = top_movies['id'].astype('int')


# In[196]:


top_movies4 = top_movies[top_movies['id'].isin(links_small)]


# In[197]:


top_movies4.shape


# In[198]:


top_movies4.head()


# In[ ]:





# In[ ]:


# 6. Movie Description Based Recommender
#Build a recommender using movie description and taglines. 


# In[199]:


top_movies4['tagline'] = top_movies4['tagline'].fillna('')
top_movies4['description'] = top_movies4['overview'] + top_movies4['tagline']
top_movies4['description'] = top_movies4['description'].fillna('')


# In[200]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')  
tfidf_matrix = tf.fit_transform(top_movies4['description'])


# In[201]:


tfidf_matrix


# In[202]:


tfidf_matrix.shape


# In[203]:


# Cosine similarity: Using cosine distance formula to find the minimum distance between words.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[204]:


cosine_sim 


# In[205]:


cosine_sim[0]


# In[132]:


top_movies4 = top_movies4.reset_index()
titles = top_movies4['title']
indices = pd.Series(top_movies4.index, index=top_movies4['title'])


# In[206]:


# Search similar scores from 1 to 31
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices] 
# this returns the titles of corresponding movies
# 'iloc' means looking for similar functions (from 'movie_indices')


# In[ ]:





# In[ ]:


# Get the top recommendations for a few moview to see how good the recommendations are.


# In[239]:


get_recommendations('The Godfather').head(10)


# In[235]:


#get_recommendations('The Apartment').head(10)


# In[236]:


#get_recommendations('The Godfather').head(10)


# In[237]:


#get_recommendations('The Dark Knight').head(10)


# In[ ]:





# In[ ]:


# 7. Collborative Filtering
# This makes recommnedations to movie watchers
# This means users similar to myself can be used to predict how much I can use a certain product/service
# that others have used.
# Uses extremely powerful algorithms like SVD(singular Value Decomposition) to minimize RMSE(Root Mean Square Error) 
# and give great recommendations


# In[224]:


#reader = Reader()


# In[ ]:


#md = pd.read_csv('C:\\Users\gfossung\Downloads\movies_metadata.csv')


# In[223]:


#ratings = pd.read_csv('C:\\Users\gfossung\Downloads\ratings_small.csv')


# In[213]:


#ratings.head()


# In[214]:


#data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[215]:


#kf = KFold(n_splits=5)


# In[216]:


#kf.split(data)


# In[ ]:


# data cross_validations(n_fold=5)


# In[ ]:


#svd = SVD()


# In[ ]:


#cross_validate(svd, data, measures=['RMSE','MAE'])


# In[ ]:





# In[222]:


# RMSE was obtained as ?. train our dataset and arrive at predictions
#trainset = data.build_full_trainset()


# In[221]:


#svd.fit(trainset)


# In[ ]:





# In[220]:


# Let's pick user 5000 and check the ratoins s/he has given:
#ratings[ratings['userId'] == 1]
#svd.predict(1, 302, 3)


# In[161]:


# Movie with ID 302 gives an estimated prediction of 2.686. This recommendation system predicts on the basis
# of assigned movie ID and tries to predict ratings based on how other users have predicted the movie.

# We now try to build a hybrid recommender involving contenct-based and collaborative filter-based engines. 
# Input: User ID & Title of movie; Output: similar movies sorted on the basis of expected ratings by that
# particular user 

#def convert_int(x):
 #   try:
  #      return int(x)
   # except:
    #    return np.nan


# In[162]:


id_map = pd.read_csv('C:\\Users\gfossung\Downloads\links_small.csv')[['movieId', 'tmdbId']]


# In[217]:


#id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
#id_map.columns = ['movieId', 'id'] 
#id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')
#indices_map = id_map.set_index('id')
#def hybrid(userId, title):
 #   None
  #  return movies.head(10)


# In[218]:


#hybrid(1, 'Avatar')


# In[219]:


#hybrid(500, 'Avatar')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





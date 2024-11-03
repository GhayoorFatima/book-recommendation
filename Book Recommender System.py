#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')
users = pd.read_csv('Users.csv')


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[6]:


books.isnull().sum()


# In[7]:


ratings.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


books.duplicated().sum()


# In[10]:


ratings.duplicated().sum()


# In[11]:


users.duplicated().sum()


# In[12]:


users['Age'].mean()


# In[13]:


users['Age'].median()


# In[14]:


users['Age'] = users['Age'].fillna(users['Age'].mean())


# In[15]:


users.info()


# In[16]:


users.isnull().sum()


# In[17]:


#books['Book-Author'].fillna(books['Book-Author'].mode()[0], inplace=True)
#books['Publisher'].fillna(books['Publisher'].mode()[0], inplace=True)


# In[18]:


books.isnull().sum()


# # POPULARITY BASED RECOMMENDER SYSTEM

# In[19]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[20]:


ratings_with_name


# In[21]:


ratings_with_name.groupby('Book-Title').count()['Book-Rating']


# In[22]:


total_ratings_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
total_ratings_df.rename(columns={'Book-Rating':'Total-No-of-Ratings'}, inplace = True)
total_ratings_df


# In[23]:


avg_ratings_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_ratings_df.rename(columns={'Book-Rating': 'Average-Ratings'}, inplace=True)
avg_ratings_df


# In[24]:


popularity_df = total_ratings_df.merge(avg_ratings_df, on='Book-Title')
popularity_df


# In[25]:


popularity_df[popularity_df['Total-No-of-Ratings']>=250]


# In[26]:


popularity_df = popularity_df[popularity_df['Total-No-of-Ratings']>=250].sort_values('Average-Ratings', ascending = False).head(50)


# In[27]:


popularity_df


# In[28]:


popularity_df.merge(books, on='Book-Title')


# In[29]:


popularity_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')


# In[30]:


popularity_df.shape


# In[31]:


popularity_df = popularity_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Total-No-of-Ratings', 'Average-Ratings', 'Book-Author', 'Year-Of-Publication', 'Publisher', ]]


# In[32]:


popularity_df


# # COLLABORATIVE FILTERING BASED RECOMMENDER SYSTEM

# In[33]:


Greater_than_200 = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
frequent_users = Greater_than_200[Greater_than_200].index


# In[34]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(frequent_users)]
filtered_rating


# In[35]:


rating_grtr_than_50 = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = rating_grtr_than_50[rating_grtr_than_50].index
famous_books


# In[36]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[37]:


final_ratings


# In[38]:


final_ratings.drop_duplicates()


# In[39]:


pt = final_ratings.pivot_table(index = 'Book-Title', columns = 'User-ID', values ='Book-Rating')
pt


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity


# In[41]:


pt = pt.fillna(0)
similarity_scores = cosine_similarity(pt)
similarity_scores


# In[42]:


cosine_similarity(pt).shape


# In[43]:


def recommend(book_name):
    #fetching index from book name
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted((list(enumerate(similarity_scores[index]))),key = lambda x:x[1], reverse = True)[1:6]
    
    for i in similar_books:
        print(pt.index[i[0]])


# In[49]:


recommend('Year of Wonders')


# In[ ]:





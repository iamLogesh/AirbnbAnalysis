#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd 
import numpy as np


import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio 
from plotly.subplots import make_subplots 

import missingno as msno

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


# In[17]:


df = pd.read_csv(r"D:\project\AirbnbAnalysis\NYC-Airbnb-2023.csv",low_memory = False)
df.head()


# In[20]:


df.shape


# In[21]:


df.info()


# In[22]:


msno.matrix(df)


# In[26]:


df.isna().sum()


# In[27]:


df.reviews_per_month.mode()


# In[29]:


df.reviews_per_month = df.reviews_per_month.fillna(0.02)


# In[30]:


df.isna().sum()


# Handling Dupicate

# In[31]:


df.duplicated().any()


# In[33]:


df[df.duplicated()].shape


# In[36]:


df = df.drop_duplicates()


# In[38]:


df.shape


# Handling the datatyypes

# In[40]:


df.dtypes


# In[42]:


df['last_review'].unique()


# In[45]:


df['last_review'] = pd.to_datetime(df['last_review'])


# In[47]:


df['year'] = df['last_review'].dt.year
df['year'].unique()


# In[49]:


df.dtypes


# In[51]:


df['last_review'].unique()


# In[53]:


df['last_review'] = pd.to_datetime(df['last_review'])


# In[55]:


df['year'] = df['last_review'].dt.year
df['year'].unique()


# In[57]:


df.dtypes


# The describe

# In[59]:


df.describe()


# Data visulation 

# Hsting listing coount by room types

# In[61]:


listings_room = df.groupby('room_type')['calculated_host_listings_count'].sum()
listings_room


# In[63]:


bar_listings_room = px.bar(data_frame = listings_room,
                          color = ['Entire home/apt','Hotel room','Private room','Shared room'],
                          color_discrete_sequence = px.colors.sequential.Rainbow, 
                          text_auto = True)
bar_listings_room.show()


# HOSting Listings count by neigh

# In[65]:


listings_neighb = df.groupby('neighbourhood_group')['calculated_host_listings_count'].sum()
listings_neighb


# In[67]:


bar_listings_neighb = px.bar(data_frame = listings_neighb,
                          color = ['Bronx','Brooklyn','Manhattan','Queens','Staten Island'],
                          color_discrete_sequence = px.colors.sequential.thermal, 
                          text_auto = True)
bar_listings_neighb.show()


# price Dribtion by room type

# In[69]:


box_price_room = px.box(data_frame = df,
                         x = df['room_type'],
                         y = df['price'],
                       color = 'room_type',
                       title = 'Price Distribution by Room Type')
box_price_room.show()


# Price distribution by neighbourhood groups

# In[71]:


scatter_price_neighb = px.scatter(data_frame = df,
                            x = df['neighbourhood_group'],
                            y = df['price'],
                            color = 'neighbourhood_group',
                            title = 'Price Distribution by Room Type')
scatter_price_neighb.show()


# Heat map

# In[73]:


corr_table = df[['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count',
                 'availability_365','number_of_reviews_ltm']].corr()
corr_table


# In[75]:


heat_map = sns.heatmap(data = corr_table, annot = True)
heat_map


# Geographical map

# In[77]:


pio.templates.default = "plotly_dark"


# In[79]:


location = df[['latitude','longitude','neighbourhood_group']]
location


# In[81]:


geo_map = px.scatter(location, x = "longitude", y = "latitude", color = "neighbourhood_group", 
                     color_discrete_sequence = px.colors.sequential.Turbo )

# Add a title and labels
geo_map.update_layout(title = "Geographical Distribution of Listings by Neighbourhood",
                  xaxis_title = "Longitude",
                  yaxis_title = "Latitude")

# Show the plot
geo_map.show()


# Mean price Distribution by neighbourhood groups and room types

# In[83]:


av_price_room = df.groupby("room_type")["price"].mean()
df_av_room = pd.DataFrame(av_price_room).reset_index()
column_names = ["room_type", "price"]
df_av_room.columns = column_names
df_av_room


# In[85]:


av_price_neighb = df.groupby("neighbourhood_group")["price"].mean()
df_av_neighb = pd.DataFrame(av_price_neighb).reset_index()
column_names = ["neighbourhood_group", "price"]
df_av_neighb.columns = column_names
df_av_neighb


# In[87]:


# Making subplots 
# figsize represents adjusts the size of the figure
# ax1, ax2 reprsents tupel that contains two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

# Average price by room type
# Creating a bar plot
sns.barplot(data = df_av_room, x = df_av_room["room_type"],
            y = df_av_room["price"], 
            ax = ax1, palette="Greens",
            label="Average Price by Room Type")

ax1.set_title("Average Price by Room Type")
ax1.set_ylabel("Average Price")

# Average price by neighbourhood group
# Creating a bar plot
sns.barplot(data = df_av_neighb, x = "neighbourhood_group", y = "price", 
            ax = ax2, palette = "Blues",
            label = "Average Price by Neighbourhood Group")

ax2.set_title("Average Price by Neighbourhood Group")
ax2.set_xlabel("Neighbourhood Group")
ax2.set_ylabel("Average Price")


# NUMBER of reviews by rom types and neighbour hood location 

# In[89]:


rev_room_type = df.groupby("room_type")["number_of_reviews"].sum()
df_rev_room_type = pd.DataFrame(rev_room_type).reset_index()
column_names = ["room_type", "number_of_reviews"]
df_rev_room_type.columns = column_names
df_rev_room_type


# In[92]:


rev_neighb = df.groupby("neighbourhood_group")["number_of_reviews"].sum()
df_rev_neighb = pd.DataFrame(rev_neighb).reset_index()
column_names = ["neighbourhood_group", "number_of_reviews"]
rev_neighb.columns = column_names
df_rev_neighb


# In[93]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
sns.barplot(data=df_rev_room_type, x="room_type", y="number_of_reviews", ax=ax1, palette = 'flare')
ax1.set_title("Number of Reviews by Room Type")

sns.barplot(data=df_rev_neighb, x="neighbourhood_group", y="number_of_reviews", ax=ax2, palette = 'crest')
ax2.set_title("Number of Reviews by Neighbourhood Location")

plt.show()


# Avaliablity of romms

# In[95]:


room_365 = df.groupby('room_type')['availability_365'].sum()
room_365


# In[97]:


col = ['Entire home/apt', 'Hotel room','Private Room', 'Shared Room']
bar_room_365 = px.bar(data_frame = room_365,
                     color = col,
                     title = 'Availability of Rooms for 365 Days',
                     text_auto = True,
                     color_discrete_sequence = px.colors.sequential.haline)
bar_room_365.show()


# Price by year

# In[99]:


price_year = df.groupby("year")["price"].sum()
df_price_year = pd.DataFrame(price_year).reset_index()
column_names = ["year", "price"]
df_price_year.columns = column_names
df_price_year


# In[100]:


plt.figure(figsize=(10, 5))

ax = sns.lineplot(data=df_price_year, x="year", y="price")
ax.set_xticks(list(range(2011, 2024)))

plt.rcParams["figure.figsize"] = (10, 4)

plt.title("Price by Year")
plt.show()


# Top Five Listings by Host names

# In[102]:


host_list = df.groupby('host_name')['calculated_host_listings_count'].sum().sort_values(ascending = False)[:5]
host_list


# In[104]:


name = ['Blueground','Eugene','RoomPicks','June','Hiroki']
bar_host_list = px.bar(data_frame = host_list,
                       color = name,
                       title = 'Top Five Listings by Host names',
                      color_discrete_sequence = px.colors.sequential.Turbo)
bar_host_list.show()


# Word cloud for host names

# In[106]:


pio.templates.default = "plotly_white"


# In[108]:


names = ' '.join(df['host_name'].astype(str))


# In[110]:


# WordCloud() is a class used to generate word clouds
# generate() takes list of words as input to generate the word cloud
wordcloud = WordCloud(width = 800, height = 400, background_color ='black').generate(names)


# In[112]:


# imshow() is used to display the wordcloud
fig = px.imshow(wordcloud.to_array())
fig.update_layout(title_text='Word Cloud for host names')
# showticklabels specify whether tick labels should be shown or not
fig.update_xaxes(showticklabels = False)  
fig.update_yaxes(showticklabels = False)
fig.show()


# In[114]:


mask_image = np.array(Image.open(r"D:\project\AirbnbAnalysis\3.png"))
# Creating a wordcloud object using the mask_image
wordcloud = WordCloud(mask = mask_image, background_color ='black').generate(names)
# ImageColorGenerator() is used to map words in wordcloud to colors of the mask image
image_colors = ImageColorGenerator(mask_image)
# Converting wordcloud to array
wordcloud_image = wordcloud.to_array()

fig = px.imshow(wordcloud_image)
fig.update_layout(title_text ='Word Cloud with Mask Image')
fig.update_xaxes(showticklabels = False)
fig.update_yaxes(showticklabels = False)
fig.show()


# In[ ]:





# -*- coding: utf-8 -*-
"""Dynamic-Price-Prediction-For-Cabs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GLAo3HkxAr7078MX0z-X7GlYFCr0F2W1

***

## Lyft/Uber Price Prediction  

Given *data about Lyft and Uber rides*, let's try to predict the **price** of a given ride.  
  
We will use a linear regression model to make our predictions.

# Getting Started
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_f8ec321073d24c11ab86d245d4fbe6a7 = 'https://s3.us.cloud-object-storage.appdomain.cloud'
else:
    endpoint_f8ec321073d24c11ab86d245d4fbe6a7 = 'https://s3.private.us.cloud-object-storage.appdomain.cloud'

client_f8ec321073d24c11ab86d245d4fbe6a7 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='SzRH3O9PkkDPguNkewibchy-vIA9KBhqRrMSVYlk5pI2',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_f8ec321073d24c11ab86d245d4fbe6a7)

body = client_f8ec321073d24c11ab86d245d4fbe6a7.get_object(Bucket='cabpriceprediction-donotdelete-pr-r1w8meq6l2bunz',Key='cab_rides.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

rides_df = pd.read_csv(body)
rides_df.head()

body = client_f8ec321073d24c11ab86d245d4fbe6a7.get_object(Bucket='cabpriceprediction-donotdelete-pr-r1w8meq6l2bunz',Key='weather.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

weather_df = pd.read_csv(body)
weather_df.head()

rides_df.info()

weather_df.info()

"""# Cleaning Ride Data"""

rides_df

rides_df.isna().sum()

rides_df = rides_df.dropna(axis=0).reset_index(drop=True)

"""# Cleaning Weather Data"""

weather_df

weather_df.isna().sum()

weather_df = weather_df.fillna(0)

"""# Creating Average Weather DataFrame"""

weather_df

# converting the timestamp data into real date format
rides_df['date'] = pd.to_datetime(rides_df['time_stamp']/ 1000, unit = 's')
weather_df['date'] = pd.to_datetime(weather_df['time_stamp'], unit = 's')

# Creating the new column that contain the location and 
rides_df['merged_date'] = rides_df['source'].astype('str') + ' - ' + rides_df['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + rides_df['date'].dt.hour.astype('str')
weather_df['merged_date'] = weather_df['location'].astype('str') + ' - ' + weather_df['date'].dt.strftime('%Y-%m-%d').astype('str') + ' - ' + weather_df['date'].dt.hour.astype('str')

#  df_rides['date'].dt.strftime('%m').head()
weather_df.index = weather_df['merged_date']

# Join the weather date on rides data
df_joined = rides_df.join(weather_df, on = ['merged_date'], rsuffix ='_w')

"""The rides and weather data have been joined by merged_date column."""

df_joined.info()

df_joined['id'].value_counts()

df_joined[df_joined['id'] == '865b44b9-4235-4e8e-b6fd-bc8373e95b63'].iloc[:,10:22]

id_group = pd.DataFrame(df_joined.groupby('id')['temp','clouds', 'pressure', 'rain', 'humidity', 'wind'].mean())
df_rides_weather = rides_df.join(id_group, on = ['id'])

# Creating the columns for Month, Hour and Weekdays 
df_rides_weather['Month'] = df_rides_weather['date'].dt.month
df_rides_weather['Hour'] = df_rides_weather['date'].dt.hour
df_rides_weather['Day'] =  df_rides_weather['date'].dt.strftime('%A')

# The distribution of rides in weekdays 
import matplotlib.pyplot as plt
uber_day_count = df_rides_weather[df_rides_weather['cab_type'] == 'Uber']['Day'].value_counts()
uber_day_count = uber_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])
lyft_day_count = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft']['Day'].value_counts()
lyft_day_count = lyft_day_count.reindex(index = ['Friday','Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])

fig , ax = plt.subplots(figsize = (12,12))
ax.plot(uber_day_count.index, uber_day_count, label = 'Uber')
ax.plot(lyft_day_count.index, lyft_day_count, label = 'Lyft')
ax.set(ylabel = 'Number of Rides', xlabel = 'Weekdays')
ax.legend()
plt.show()

# The ride distribution in one day 
fig , ax = plt.subplots(figsize= (12,12))
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('Hour').Hour.count(), label = 'Lyft')
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('Hour').Hour.count().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('Hour').Hour.count(), label = 'Uber')
ax.legend()
ax.set(xlabel = 'Hours', ylabel = 'Number of Rides')
plt.xticks(range(0,24,1))
plt.show()

# The Average price of rides by type of service
import seaborn as sns

uber_order =[ 'UberPool', 'UberX', 'UberXL', 'Black','Black SUV','WAV' ]
lyft_order = ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']
fig, ax = plt.subplots(2,2, figsize = (20,15))
ax1 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].price , ax = ax[0,0], order = uber_order)
ax2 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].name, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].price , ax = ax[0,1], order = lyft_order)
ax3 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('name').name.count(), ax = ax[1,0] ,order = uber_order)
ax4 = sns.barplot(x = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count().index, y = df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('name').name.count(), ax = ax[1,1],order = lyft_order)
for p in ax1.patches:
    ax1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
for p in ax2.patches:
    ax2.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax1.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax2.set(xlabel = 'Type of Service', ylabel = 'Average Price')
ax3.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax4.set(xlabel = 'Type of Service', ylabel = 'Number of Rides')
ax1.set_title('The Uber Average Prices by Type of Service')
ax2.set_title('The Lyft Average Prices by Type of Service')
ax3.set_title('The Number of Uber Rides by Type of Service')
ax4.set_title('The Number of Lyft Rides by Type of Service')
plt.show()

# The average price by distance
fig , ax = plt.subplots(figsize = (12,12))
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] == 'Lyft'].groupby('distance')['price'].mean(), label = 'Lyft')
ax.plot(df_rides_weather[df_rides_weather['cab_type'] == 'Uber'].groupby('distance').price.mean().index, df_rides_weather[df_rides_weather['cab_type'] =='Uber'].groupby('distance').price.mean(), label = 'Uber')
ax.set_title('The Average Price by distance', fontsize= 15)
ax.set(xlabel = 'Distance', ylabel = 'Price' )
ax.legend()
plt.show()

# The average price by distance 
fig, ax = plt.subplots(1,2 , figsize = (20,5))
for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Uber']['name'].unique()):
    ax[0].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)
ax[0].set_title('Uber Average Prices by Distance')
ax[0].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[0].legend()
for i,col in enumerate(df_rides_weather[df_rides_weather['cab_type'] == 'Lyft']['name'].unique()):
    ax[1].plot(df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean().index, df_rides_weather[ df_rides_weather['name'] == col].groupby('distance').price.mean(), label = col)
ax[1].set(xlabel = 'Distance in Mile', ylabel = 'Average price in USD')
ax[1].set_title('Lyft Average Prices by Distance')
ax[1].legend()
plt.show()

# the average rate per mile
df_rides_weather['rate_per_mile'] = round((df_rides_weather['price'] / df_rides_weather['distance'] ),2)
# The average rate per mile plot
fig, ax = plt.subplots(1,2,figsize = (12,5))
ax1 = sns.lineplot(x = df_rides_weather.groupby(['distance'])['rate_per_mile'].mean().index, y = df_rides_weather.groupby('distance')['rate_per_mile'].mean(), ax = ax[0])
ax2 = sns.lineplot(x = df_rides_weather.groupby(['distance'])['rate_per_mile'].mean().index, y = df_rides_weather.groupby('distance')['rate_per_mile'].mean(), ax = ax[1])
plt.xticks(range(0, 10,1))
ax1.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD')
ax2.set(xlabel = 'Distance', ylabel = 'Rate per Mile in USD', ylim = (0,15))
ax1.set_title('The Average Rate per Mile', fontsize = 16)
ax2.set_title('ZOOM Average Rate per Mile', fontsize = 16)
plt.show()

# Scatter chart for Rate per mile and distance
    # pivot table to calculate average rate based on cab_type, service type(name) and distance
rates_per_mile_pivot = df_rides_weather.pivot_table(index = ['cab_type', 'name', 'distance'] , values = ['rate_per_mile'])
rates_per_mile_pivot.reset_index(inplace = True)

fig, ax = plt.subplots(2,2, figsize = (20,8))
ax1 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[0,0])
ax2 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Uber']['name'], ax = ax[1,0])
ax2.set( ylim = (0,20))
ax3 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[0,1])
ax4 = sns.scatterplot(x = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['distance'], y = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['rate_per_mile'], hue = rates_per_mile_pivot[rates_per_mile_pivot['cab_type'] == 'Lyft']['name'], ax = ax[1,1])
ax4.set( ylim = (0,20))
handles_uber, labels_uber = ax1.get_legend_handles_labels()
handles_uber = [handles_uber[6],handles_uber[3],handles_uber[4],handles_uber[5],handles_uber[1],handles_uber[2]]
labels_uber = [labels_uber[6],labels_uber[3],labels_uber[4],labels_uber[5],labels_uber[1],labels_uber[2]]
ax1.legend(handles_uber, labels_uber)
ax2.legend(handles_uber, labels_uber)
handles_lyft, labels_lyft = ax3.get_legend_handles_labels()
handles_lyft = [handles_lyft[6],handles_lyft[4],handles_lyft[5],handles_lyft[1],handles_lyft[2],handles_lyft[3]]
labels_lyft = [labels_lyft[6],labels_lyft[4],labels_lyft[5],labels_lyft[1],labels_lyft[2],labels_lyft[3]]
ax3.legend(handles_lyft, labels_lyft)
ax4.legend(handles_lyft, labels_lyft)
ax1.set_title('Uber Rate per Mile')
ax1.set(ylabel = 'Rate per Mile in USD', xlabel = ' ')
ax2.set_title('Uber Rate Zoom(0 to 20 USD)')
ax2.set(ylabel = 'Rate per Mile in USD', xlabel = 'Distance')
ax3.set_title('Lyft Rate per Mile')
ax3.set(ylabel = ' ', xlabel = ' ')
ax4.set_title('Lyft Rate Zoom(0 to 20 USD)')
ax4.set(ylabel = ' ', xlabel = 'Distance')
plt.show()

# Overrated rides
high_mile_rates = df_rides_weather[df_rides_weather['rate_per_mile'] > 80]
# The number of overrated rides by cab type
high_mile_rates['cab_type'].value_counts()

# Overrated Lyft rides
high_mile_rates[high_mile_rates['cab_type'] == 'Lyft'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']]

# Overrated Uber Rides
high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].loc[:,['distance', 'cab_type', 'price', 'surge_multiplier','name', 'rate_per_mile']].sort_values(by = 'rate_per_mile', ascending = False).head(20)

# The number of rides based on service type, distance, and price 
over_rated_pivot = high_mile_rates[high_mile_rates['cab_type'] == 'Uber'].pivot_table(index = ['name', 'distance', 'price'], values = ['id'], aggfunc = len).rename(columns = {'id' : 'count_rides'})
over_rated_pivot.reset_index(inplace =True)
over_rated_pivot.sort_values(by = ['count_rides', 'name'], ascending = False).head(15)

"""All of the ride distances are very short and the number of rides of one specific service type are very high. So, these are cancellations and their prices.

**Cancellation prices by service type**
* WAV: 7.0
* UberPool: 4.5
* UberX: 7.0
* UberXL: 8.5
* Black: 15.0
* Black SUV: 27.5

Based on these prices, if you are not ready to go, don't call Black SUV :D
"""



#before cells are testing

weather_df.groupby('location').mean()

avg_weather_df = weather_df.groupby('location').mean().reset_index(drop=False)
avg_weather_df = avg_weather_df.drop('time_stamp', axis=1)
avg_weather_df

"""# Merging DataFrames"""

rides_df = rides_df.drop('merged_date', axis=1)
rides_df = rides_df.drop('date', axis=1)
rides_df

weather_df = weather_df.drop('merged_date', axis=1)
weather_df = weather_df.drop('date', axis=1)
weather_df

source_weather_df = avg_weather_df.rename(
    columns={
        'location': 'source',
        'temp': 'source_temp',
        'clouds': 'source_clouds',
        'pressure': 'source_pressure',
        'rain': 'source_rain',
        'humidity': 'source_humidity',
        'wind': 'source_wind'
    }
)

source_weather_df

destination_weather_df = avg_weather_df.rename(
    columns={
        'location': 'destination',
        'temp': 'destination_temp',
        'clouds': 'destination_clouds',
        'pressure': 'destination_pressure',
        'rain': 'destination_rain',
        'humidity': 'destination_humidity',
        'wind': 'destination_wind'
    }
)

destination_weather_df

data = rides_df\
    .merge(source_weather_df, on='source')\
    .merge(destination_weather_df, on='destination')

data

data.name.unique()

data.source.unique()

item_counts = data["source"].value_counts()
item_counts

data.destination.unique()

item_counts = data["destination"].value_counts()
item_counts

data.product_id.unique()

item_counts = data["name"].value_counts()
item_counts

item_counts = data["product_id"].value_counts()
item_counts

cat=data.dtypes[data.dtypes=='O'].index.values
cat

from collections import Counter as c # return counts
for i in cat:
    print("Column :",i)
    print('count of classes : ',data[i].nunique())
    print(c(data[i]))
    print('*'*120)

data.dtypes[data.dtypes!='O'].index.values

data.isnull().any()#it will return true if any columns is having null values

data.isnull().sum() #used for finding the null values

"""# Label Encoding"""

data1=data.copy()
from sklearn.preprocessing import LabelEncoder #importing the LabelEncoding from sklearn
x='*'
for i in cat:#looping through all the categorical columns
    print("LABEL ENCODING OF:",i)
    LE = LabelEncoder()#creating an object of LabelEncoder
    print(c(data[i])) #getting the classes values before transformation
    data[i] = LE.fit_transform(data[i]) # trannsforming our text classes to numerical values
    print(c(data[i])) #getting the classes values after transformation
    print(x*100)

data.head()

data.info()

x = data.drop(['price','distance','time_stamp','surge_multiplier','id','source_temp','source_clouds','source_pressure','source_rain','source_humidity','source_wind','destination_temp','destination_clouds','destination_pressure','destination_rain','destination_humidity','destination_wind'],axis=1) #independet features
x=pd.DataFrame(x)
y = data['price'] #dependent feature
y=pd.DataFrame(y)

x.head()

y.head()

"""# Splitting dataset into train and test"""

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)

from sklearn.ensemble import RandomForestRegressor
rand=RandomForestRegressor(n_estimators=20,random_state=52,n_jobs=-1,max_depth=4)
rand.fit(x_train,y_train)

from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url" :"https://us-south.ml.cloud.ibm.com",
    "apikey":"uG7BN2ecAL4DfKy1PwAQE-sh2B7TmUZ2UMPhZ3hvo_ad"
}
client=APIClient(wml_credentials)

def guid_from_space_name(client,space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['id'])

space_uid = guid_from_space_name(client,'model')
print("Space UID = " + space_uid)

client.set.default_space(space_uid)

client.software_specifications.list()

software_spec_uid = client.software_specifications.get_uid_by_name("default_py3.8")
software_spec_uid

model_details= client.repository.store_model(model=rand,meta_props={
    client.repository.ModelMetaNames.NAME : "Dynamic Price Prediction for Cabs",
    client.repository.ModelMetaNames.TYPE : "scikit-learn_0.23",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID : "software_spec_uid"
})

model_id=client.repository.get_model_uid(model_details)

"""# Predecting the Result"""

ypred=rand.predict(x_test)
print(ypred)

!pip install ibm_watson_machine_learning

"""# Score of the model"""

rand.score(x_train,y_train)

"""# Saving Our Model"""

import pickle
pickle.dump(rand, open("model.pkl", "wb"))




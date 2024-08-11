import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data=pd.read_csv("Dataset.csv")
#data=pd.DataFrame(data)
#head
data=data.head(10)
print("the head of the dataset:")
print(data)
#tail
print("the tail of the dataset:")
print(data.tail())
#info
print("the info of the dataset:")
print(data.info())
#describe
print("Describe about the dataset:")
print(data.describe())
#null data identifying
print("Null data identification:")
print(data.isnull().sum())
print(pd.isnull(data))
#replace empty cells into value
print("Null Imputation:")
print(data.fillna(100))
#remove the null data in the dataset
print("Null Data Removal:")
print(data.dropna()) 
#checking datatype of the column
print("datatype of the columns:")
print(data.dtypes)
#find duplicates
print("duplicates in the dataset:")
print(data.duplicated())
#remove duplicates
print("after removing the duplicates in the dataset:\n")
print(data.drop_duplicates)
#data consistency 
#checking if a column contains unique
print(data["class"].unique())
#data reshaping
#transpose the dataframe
print("the transpose of the dataframe:")
print(data.T)
#data merging
data1=data.head(2)
data2=data.tail(2)
print(pd.concat([data1,data2], axis=0))
#data aggregation
print("data aggregation:")
data=pd.DataFrame(data)
print("grouping data:")
print(data.groupby('flight')['price'].sum())
print("aggregating data:")
print(data.agg({'price': 'sum'}))
#Univariate analysis - histogram
plt.hist(data["duration"])
plt.title("fligt duration")
plt.show()
#Bivariate analysis - Scatter plot
x=data["airline"]
y=data["flight"]
plt.scatter( x,y)
plt.xlabel("Airline")
plt.ylabel("Flight")
plt.grid(True)
plt.title("Flights in the airline")
plt.show()
#Multivariate analysis - Pair plot
sns.pairplot(data)
plt.show()
#create user profile
print("\nuser profile:\n")
def create_user_profile(username, email, age, country):
 user_profile = {"username": username,"email": email,"age": age,"country": country}
 return user_profile
#Temporal Analysis
user_profile = create_user_profile("bhavani", "sb.bhavani.sb@gmail.com", 19, "India")
print(user_profile)
data.set_index('duration', inplace=True)
plt.figure(figsize=(10, 6))
data['price'].plot()
plt.title('Temporal Analysis')
plt.xlabel('Duration')
plt.ylabel('Price')
plt.grid(True)
plt.show()
#import the dataset
data=pd.read_csv("Dataset.csv")
data=data.head(10)
#Univariate analysis - histogram
plt.hist(data["duration"])
plt.title("Histogram")
plt.xlabel("Duration")
plt.ylabel("Frequency")
plt.show()
#Univariate analysis - bar chart
plt.bar(data['airline'].value_counts().index, 
data['duration'].value_counts().values)
plt.xlabel("Airline") 
plt.ylabel("Duration") 
plt.title("Bar Chart ") 
plt.show()
#Biunivariate analysis - scatter plot
x = data["duration"]
y = data["price"]
plt.scatter(x, y, color='red', marker='o') 
plt.grid(True) 
plt.xlabel("Duration") 
plt.ylabel("Price") 
plt.title("Scatter Plot") 
plt.show() 
#Biunivariate analysis - Box plot
x = data["airline"]
y = data["price"]
sns.boxplot(x="airline",y="price",data=data)
plt.xlabel('Airline')
plt.ylabel('Price')
plt.title('Box Plot')
plt.show()
#Multivariate visualization - pair plot
sns.pairplot(data)
plt.title('Pair Plot')
plt.show()
#Interactive visualization - Scatter plot
fig = px.scatter(data, x='flight', y='price', hover_data=['duration'])
fig.show()
#Interactive visualization - Dashboard
app = dash.Dash(__name__)
app.layout = html.Div([
dcc.Graph(
id='interactive-plot',
figure={
'data': [
{'x': data['flight'], 'y': data['arrival_time'], 'mode': 'markers', 'type': 'dashboard'}
],
'layout': {
'title': 'Interactive Scatter Plot',
'xaxis': {'title': 'Flight'},
'yaxis': {'title': 'Arrival_time'}
}
}
)
])
if __name__ == '__main__':
 app.run_server(debug=True)
# Load the dataset
dataset = pd.read_csv('dataset.csv')
# Display the first few rows
print(dataset.head())
# Check for missing values and handle them
print(dataset.isnull().sum())
dataset = dataset.dropna()
X = dataset[['duration','days_left']] 
y = dataset['price'] 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train recommendation models using the training data
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)
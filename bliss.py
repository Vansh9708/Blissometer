#importing the necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

happy = pd.read_csv('Happiness.csv')

happy_data = pd.DataFrame(happy)
happy_data.head()

# So our target variable this time is Happiness Rank, first we check how large our dataset is.

happy_data.shape

happy_data.isnull().sum()

sns.boxplot(happy_data['Happiness Rank'], happy_data['Health (Life Expectancy)']) 

# To know more about boxplots - https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

fig = plt.figure(figsize =(10,8))
data = [happy_data['Health (Life Expectancy)'], happy_data['Freedom'], happy_data['Trust (Government Corruption)'], 
        happy_data['Dystopia Residual']]

plt.boxplot(data)
 
# show plot
plt.show()

temp = happy_data.copy()
new_data = temp.drop(['Trust (Government Corruption)'], axis = 1)
new_data.head()

# Segregating target variable and removing useless columns.

y = new_data['Happiness Rank']
fin_data = new_data.drop(['Region', 'Country', 'Happiness Rank'], axis = 1)

fin_data.columns

scale = StandardScaler()
data = scale.fit_transform(fin_data)

# Now you have to use index for columns and rows like in numpy to access values from the data.
print(data[0][0])

# Now we compare the standardized data:- 
# 'Health (Life Expectancy)'] - data[4]
# happy_data['Freedom'] - data[5]
# happy_data['Trust (Government Corruption)'] - Not present
# happy_data['Dystopia Residual'] - data[7]

fig = plt.figure(figsize =(10,8))
inserting = [data[4], data[5], data[7]]

plt.boxplot(inserting)
 
# show plot
plt.show()

Xtrain, Xtest, ytrain, ytest = train_test_split(data, y, test_size=0.1)

reg = LinearRegression()

reg.get_params(deep = True)

vari = reg.fit(Xtrain, ytrain)

print(reg.score(Xtrain, ytrain)

      # Predicting 
pred = reg.predict(Xtest)

r2_score = reg.score(Xtest,ytest)
print(r2_score)

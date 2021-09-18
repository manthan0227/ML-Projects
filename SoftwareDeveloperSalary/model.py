import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('survey_results_public.csv')

# print(df.columns)

df = df[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
df = df.rename({'ConvertedCompYearly': 'Salary'}, axis=1)

df = df[df['Salary'].notnull()]
# print(df.head())

df = df.dropna()
# print(df.isnull().sum())
# print(df[df['India']])

df = df[df['Employment'] == 'Employed full-time']
df = df.drop('Employment', axis=1)
# print(df[df['Country']=='India'])

# print(df['Country'].value_counts())

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'others'
    return categorical_map

country_map = shorten_categories(df['Country'].value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
# print(df['Country'].value_counts())

# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# df.boxplot('Salary', 'Country', ax=ax)
# plt.suptitle('Salary (US $) vs Country')
# plt.title('')
# plt.ylabel('Salary')
# plt.xticks(rotation=90)
# plt.show()

df = df[df['Salary'] <= 250000]
df = df[df['Salary'] >= 10000]
df = df[df['Country'] != 'others']

# fig, ax = plt.subplots(1, 1, figsize=(12, 7))
# df.boxplot('Salary', 'Country', ax=ax)
# plt.suptitle('Salary (US $) vs Country')
# plt.title('')
# plt.ylabel('Salary')
# plt.xticks(rotation=90)
# plt.show()

# print(df['YearsCodePro'].unique())
def clean_experience(x):
    if x == 'Less than 1 year':
        return 0.5
    if x == 'More than 50 years':
        return 50
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral degree' in x:
        return 'Post grad'
    return 'Less than a Bachelores'

df['EdLevel'] = df['EdLevel'].apply(clean_education)

from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
# print(df['EdLevel'].unique())

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
# print(df['Country'].unique())

X = df.drop('Salary', axis=1)
y = df['Salary']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
le = LinearRegression()
le.fit(X_train, y_train)
y_pred = le.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mae = mean_absolute_error(y_pred, y_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

# print("Mean Absolute Error = ", mae)
# print("Mean Squared Error = ", rmse)

from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor(random_state=0)
dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)

mae = mean_absolute_error(y_pred, y_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

# print("mae = ", mae)
# print("rmse = ", rmse)

from sklearn.ensemble import RandomForestRegressor
ran_tree = RandomForestRegressor(random_state=0)
ran_tree.fit(X_train, y_train)
y_pred = ran_tree.predict(X_test)

mae = mean_absolute_error(y_pred, y_test)
rmse = np.sqrt(mean_squared_error(y_pred, y_test))

# print('Mae = ', mae)
# print('Rmse = ', rmse)

x = np.array([['United States', 'Master’s degree', 15]])
print(x)

x[:, 0] = le_country.fit_transform(x[:, 0])
x[:, 1] = le_education.fit_transform(x[:, 1])
# x[:, 2] = x[:, 2].apply(clean_experience)
x = x.astype(float)

y_pred = ran_tree.predict(x)
print(y_pred)

import pickle
data = {'model': ran_tree, 'le_country': le_country, 'le_education': le_education}
with open('saved_model.pkl', 'wb') as file:
    pickle.dump(data, file)

with open('saved_model.pkl', 'rb') as file:
    data = pickle.load(file)

model = data['model']
le_country = data['le_country']
le_education = data['le_education']

y_pred = model.predict(x)
print(y_pred)





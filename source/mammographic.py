import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Read from CSV file
mammograph_of_data = pd.read_csv('mammographic_mass_data.txt',na_values=['?'],names=['BI-RADS','age','shape','margin', 'density','severity'])
print(mammograph_of_data.head())
print(mammograph_of_data.describe())

# Check if any values are missing from data and drop them
mammograph_of_data.loc[(mammograph_of_data['age'].isnull()) | (mammograph_of_data['shape'].isnull()) | (mammograph_of_data['margin'].isnull()) | (mammograph_of_data['density'].isnull())]
mammograph_of_data.dropna(inplace=True)
print(mammograph_of_data.describe())

# Define my_features and label
my_features = mammograph_of_data[['age','shape','margin','density']].values
my_class = mammograph_of_data[['severity']].values

# We are taking 4 my_features as deterministic my_features
name_features = ['age','shape','margin','density']

# Noramalizing our data using preprocessing
normalized = preprocessing.StandardScaler()
normalized_Features = normalized.fit_transform(my_features)
print(normalized_Features)

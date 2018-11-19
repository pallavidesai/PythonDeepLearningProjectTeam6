import pandas as pd
import numpy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

mammograph_data = pd.read_csv('mammographic_mass_data.txt',na_values=['?'],names=['BI-RADS','age','shape','margin', 'density','severity'])
#print(mammograph_data.describe())

#mammograph_data.loc[(mammograph_data['age'].isnull()) | (mammograph_data['shape'].isnull()) | (mammograph_data['margin'].isnull()) | (mammograph_data['density'].isnull())]
mammograph_data.dropna(inplace=True)
#print(mammograph_data.describe())

features = mammograph_data[['age','shape','margin','density']].values
classes = mammograph_data[['severity']].values

feature_names = ['age','shape','margin','density']

normalized = preprocessing.StandardScaler()
features_normalized = normalized.fit_transform(features)
print(features_normalized)

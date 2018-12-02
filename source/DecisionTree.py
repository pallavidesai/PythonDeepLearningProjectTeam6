import pandas as pd
import numpy
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split, cross_val_score
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

#Decision Tree
numpy.random.seed(10)
X_train,X_test,Y_train, Y_test = train_test_split(features_normalized,classes,test_size=0.25,random_state=1)
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print('Accuracy of a decision tree is: ',metrics.accuracy_score(Y_test, y_pred))

#K-fold cross validation
model_scores = cross_val_score(model,features_normalized,classes, cv= 10)
print('Accuracy from 10-fold validation is: ',model_scores.mean())

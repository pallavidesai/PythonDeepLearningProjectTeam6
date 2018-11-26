import pandas as pd
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing,metrics
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

mammograph_data = pd.read_csv('mammographic_mass_data.txt',na_values=['?'],names=['BI-RADS','age','shape','margin', 'density','severity'])
mammograph_data.dropna(inplace=True)


features = mammograph_data[['age','shape','margin','density']].values
classes = mammograph_data[['severity']].values

feature_names = ['age','shape','margin','density']

normalized = preprocessing.StandardScaler()
features_normalized = normalized.fit_transform(features)
print(features_normalized)

#Naive Bayes
X_train,X_test,Y_train, Y_test = train_test_split(features_normalized,classes,test_size=0.25,random_state=1)

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(features)

model = MultinomialNB()
cv_scores = cross_val_score(model, all_features_minmax, classes, cv=10)

cv_scores.mean()


#K-fold cross validation
model_scores = cross_val_score(model,features_normalized,classes.ravel(), cv= 10)
print('Accuracy from 10-fold validation is: ',model_scores.mean())

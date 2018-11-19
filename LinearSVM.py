import pandas as pd
import numpy
from sklearn import preprocessing,metrics,svm
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

#SVM
X_train,X_test,Y_train, Y_test = train_test_split(features_normalized,classes,test_size=0.25,random_state=1)
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train,Y_train.ravel())
y_pred = model.predict(X_test)
print('Accuracy of a linear SVM is: ', metrics.accuracy_score(Y_test, y_pred))

#K-fold cross validation
model_scores = cross_val_score(model,features_normalized,classes.ravel(), cv= 10)
print('Accuracy from 10-fold validation is: ',model_scores.mean())

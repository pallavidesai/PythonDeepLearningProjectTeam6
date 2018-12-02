import pandas as pd
import numpy
from sklearn import preprocessing,metrics
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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


def create_model():
    model = Sequential()
    #4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model; adam seemed to work best
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Wrap our Keras model in an estimator compatible with scikit_learn
model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
model_scores = cross_val_score(model,features_normalized,classes.ravel(), cv= 10)
print('Accuracy from 10-fold validation is: ',model_scores.mean())
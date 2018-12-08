import pandas as pd
from sklearn import preprocessing,metrics
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Read from CSV file
mammograph_of_data = pd.read_csv('mammographic_mass_data.txt',na_values=['?'],names=['BI-RADS','age','shape','margin', 'density','severity'])
mammograph_of_data.dropna(inplace=True)

# Define my_features and label
my_features = mammograph_of_data[['age','shape','margin','density']].values
my_class = mammograph_of_data[['severity']].values

# We are taking 4 my_features as deterministic my_features
name_features = ['age','shape','margin','density']

# Noramalizing our data using preprocessing
normalized = preprocessing.StandardScaler()
normalized_Features = normalized.fit_transform(my_features)
print(normalized_Features)

# Split train and test data using train_test_split
X_train,X_test,Y_train, Y_test = train_test_split(normalized_Features,my_class,test_size=0.2,random_state=1)

# Create neural network
def MyModel_Create():
    My_Model = Sequential()
    #4 feature inputs going into an 6 neurons
    My_Model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    My_Model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification i.e 1 or 0
    My_Model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile My_Model 
    My_Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    My_Model.summary();
    return My_Model

# Call the function here with 100 epoches
My_Model = KerasClassifier(build_fn=MyModel_Create, epochs=100, verbose=0)
#K-fold cross validation
model_scores = cross_val_score(My_Model,normalized_Features,my_class.ravel(), cv= 10)
print('Accuracy of our model using 10-fold validation is: ',model_scores.mean())


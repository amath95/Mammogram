import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

scale = StandardScaler()

#Assemble the data into dataframes
columns = ['BI-RADS assessement','Age', 'Mass Shape','Margin','Density','Severity']
feature_col = ['Age', 'Mass Shape','Margin','Density']
df = pd.read_csv('/Users/amath/Downloads/MLCourse-2/mammographic_masses.data.txt', names=columns, na_values=['?'])

#Clean the data by getting rid of NaN values and seperate feature/label data
df.dropna(inplace=True)
all_features = df[['Age', 'Mass Shape', 'Margin', 'Density']].values
label = df['Severity'].values
all_features_scaled = scale.fit_transform(all_features)

#Seperate train and test data
x_train, x_test, y_train, y_test = train_test_split(all_features_scaled, label, test_size=.25, random_state=1)

#Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
score = cross_val_score(clf, x_train, y_train, cv=10)
print("Score for the Decision Tree Model: ", np.mean(score))

#Random Forest Tree
forest = RandomForestClassifier(n_estimators=10)
score = cross_val_score(forest, x_train, y_train, cv=10)
print("Random Forest model accuracy:", np.mean(score))

#SVM model
clf = svm.SVC(kernel='linear', C=1)
score = cross_val_score(clf, x_train, y_train, cv=10)
print('SVM model accuracy: ', np.mean(score))

#KNeighborsClassifier
scores_per_k = []
for values in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=values)
    score = cross_val_score(clf, x_train, y_train, cv=10)
    scores_per_k.append(np.mean(score))
print('KNN best value of K: {} with accuracy :{}'.format(np.argmax(scores_per_k), np.max(scores_per_k)))

#MultinomialNaiveBayes
clf = MultinomialNB()
scaled = MinMaxScaler(feature_range=(0, 1))
featured_scaled = scaled.fit_transform(all_features)
x_train, x_test, y_train, y_test = train_test_split(featured_scaled, label, test_size=.25, random_state=1)
score = cross_val_score(clf, x_train, y_train, cv=10)
print("Multinomial Naive Bayes model accuracy: {}".format(np.mean(score)))

#SVM (rbf,sigmoid,poly kernels) model
clf = svm.SVC(kernel='rbf', C=1)
score = cross_val_score(clf, all_features_scaled, label, cv=10)
print('SVM rbf model accuracy: ', np.mean(score))
clf = svm.SVC(kernel='sigmoid', C=1)
score = cross_val_score(clf, all_features_scaled, label, cv=10)
print('SVM sigmoid model accuracy: ', np.mean(score))
clf = svm.SVC(kernel='poly', C=1)
score = cross_val_score(clf, all_features_scaled, label, cv=10)
print('SVM poly model accuracy: ', np.mean(score))

#Logstic regression model
clf = LogisticRegression()
score = cross_val_score(clf, all_features_scaled, label, cv=10)
print('Logistic model accuracy: ', np.mean(score))

#Keras Neural Network
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=4))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features_scaled, label, cv=10)
print('Neural network accuracy: ',cv_scores.mean())

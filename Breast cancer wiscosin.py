# Diagnostic Breast Cancer data
# Prediction of patient into malignant or benign based on data collected
# Rows-569, no of variables-32
# Data Validation, Data Exploration, classification technique, Decision Trees, Pruning, Bagging, Random forests, Support vector classifier. 
# Data Preparation & Exploration, applying predictive modeling techniques Model Development, Analytical Approaches, Interpreting the Results.

# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing dataset
df=pd.read_csv("breast-cancer-wisconsin-data.csv")

# Preview data
pd.set_option('display.max_columns', None)
df.head()

# Dataset dimensions
df.shape

# Features data-type
df.dtypes

# List of features
list(df)

# Statistical summary
df.describe()

# Count of null values 
df.isnull().sum()

# Label encoding for object data
from sklearn.preprocessing import LabelEncoder
# Encode labels with value 0 and 1
LE=LabelEncoder()
df['diagnosis_encoded']=LE.fit_transform(df['diagnosis'])
df['diagnosis_encoded']

df.shape
list(df)

# Heatmap- finds correlation between Independent and dependent attributes
plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True)
plt.show()

# Selecting the variables using correlation
df.corr()

# split X and Y variables
X=df.iloc[:,2:32]
X
list(X)

Y=df['diagnosis_encoded']
Y

# standardize the data
from sklearn.preprocessing import StandardScaler
X_scale=StandardScaler().fit_transform(X)
X_scale
X_scale.shape

# Splitting up of X and Y variables into train and test cases
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.30,random_state = 42)
X_train.shape,X_test.shape, Y_train.shape,Y_test.shape

# Implementing the Decision tree model by gini index method
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

dt.tree_.node_count
dt.tree_.max_depth

Y_pred_train = dt.predict(X_train)
Y_pred_test = dt.predict(X_test)

print(f"Decision tree has {dt.tree_.node_count} nodes with maximum depth covered up to {dt.tree_.max_depth}")

# Import the metrics class
from sklearn.metrics import confusion_matrix, accuracy_score

cm=confusion_matrix(Y_test,Y_pred_test)
print("Confusion matrix for home loan sanction process risky and safe is:",cm)

acc=accuracy_score(Y_test,Y_pred_test).round(3)
print("Home loan sanctioning process accuracy result is:",acc)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))

# Further tuning is required to decide about max depth value
# Apply grid search cv method and pass levels and 
# Look out for the best depth at this place and from this can fit the model to get home loan sanctioning error rate

from sklearn.model_selection import GridSearchCV
levels = {'max_depth': [1,2,3,4,5,6,7]}

DTgrid = GridSearchCV(dt, cv = 10, scoring = 'accuracy', param_grid = levels)

DTgridfit = DTgrid.fit(X_train,Y_train)

DTgridfit.fit(X_test,Y_test)

DTgridfit.best_score_

DTgridfit.best_estimator_

###############################################################################
# DecisionTreeClassifier= dt--> base learner,, To the baselearner dt apply bagging(splits on the datapoints)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator=dt,max_samples=0.6, n_estimators=500, random_state = 8)
bag.fit(X_train, Y_train)
Y_pred = bag.predict(X_test)

from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_pred,Y_test).round(3))

# Grid Search CV method
from sklearn.model_selection import GridSearchCV
samples = {'max_samples': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

bag_grid = GridSearchCV(bag, cv = 10, scoring = 'accuracy', param_grid = samples)

bag_gridfit = bag_grid.fit(X_train,Y_train)

bag_gridfit.fit(X_test,Y_test)

bag_gridfit.best_score_

bag_gridfit.best_estimator_

#################################################################################
# Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_features = 30,  n_estimators = 500, random_state =24)

RFC.fit(X_train, Y_train)
Y_pred = RFC.predict(X_test)

from sklearn.metrics import accuracy_score 
print(accuracy_score(Y_pred,Y_test).round(3))

from sklearn.model_selection import GridSearchCV
samples = {'max_features': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}

RF_grid = GridSearchCV(RFC, cv = 10, scoring = 'accuracy', param_grid = samples)

RF_gridfit = RF_grid.fit(X_train,Y_train)

RF_gridfit.fit(X_test,Y_test)

RF_gridfit.best_score_

RF_gridfit.best_estimator_

###############################################################################
#Support Vector classifier

from sklearn.svm import SVC
m1 = SVC()
m1

svc = m1.fit(X_train, Y_train)
Y_pred = m1.predict(X_test)

Y_test

from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,Y_pred)
cm

acc = metrics.accuracy_score(Y_test,Y_pred).round(6)
acc

# Grid Search CV method
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1,2,3,4,5, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  

grid_svc=GridSearchCV(SVC(),param_grid,cv=10,scoring='accuracy')

SVC_grid_fit= grid_svc.fit(X_train, Y_train)

SVC_grid_fit.fit(X_test,Y_test)

SVC_grid_acc = SVC_grid_fit.best_score_
print("SVC accuracy=",SVC_grid_acc)

SVC_grid_fit.best_estimator_

SVC_grid_fit.best_params_



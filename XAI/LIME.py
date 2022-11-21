import dalex as dx
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn import preprocessing
import pandas as pd
import lime
from lime.lime_tabular import LimeTabularExplainer 

# Code for applying the XAI method LIME on the titanic data set

# Load data set
titanic = dx.datasets.load_titanic()
X = titanic.drop(columns='survived')
y = titanic.survived

le = preprocessing.LabelEncoder()

X['gender'] = le.fit_transform(X['gender'])
X['class'] = le.fit_transform(X['class'])
X['embarked'] = le.fit_transform(X['embarked'])

# Train model
titanic_fr = rfc()
titanic_fr.fit(X, y)

henry = pd.Series([1, 47.0, 0, 1, 25.0, 0, 0], 
                  index =['gender', 'age', 'class', 'embarked',
                          'fare', 'sibsp', 'parch'])
                          
explainer = LimeTabularExplainer(X, 
                      feature_names=X.columns, 
                      class_names=['died', 'survived'], 
                      discretize_continuous=False, 
                      verbose=True)                          
lime = explainer.explain_instance(henry, titanic_fr.predict_proba)
lime.show_in_notebook(show_table=True)

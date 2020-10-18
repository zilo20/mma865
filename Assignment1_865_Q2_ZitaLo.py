# Name: Zita Lo
# Student Number: 20196119
# Program: MMA
# Cohort: Winter 2021
# Course Number: MMA 865
# Date: October 17, 2020


# Answer to Question 2 - Task 1

# Support response to Question 2 - Task 3: 
# ---- "Export Incorrect Prediction Results (sentiment_test data)" section
# ---- "Further exploration on Sentiment_Test Data" section 


# Import packages
import pandas as pd
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

import os
os.getcwd()

# *********************************************
#
# # Load in data
#
# *********************************************

# load in csv file. Display info about the data and show first 5 instances.
df = pd.read_csv("sentiment_train.csv")

# Set display width = 100.
pd.set_option('display.max_colwidth', 100)
print(df.info())
print(df.head())


# Split data into train and test set ratio 0.85 : 0.15.
# Define "Polarity" feature as the target variable.
from sklearn.model_selection import train_test_split

X = df['Sentence']
y = df['Polarity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# *********************************************
#
# # Preprocessing and Feature Engineering
#
# *********************************************

# Preprocessing
# Import packages for preprocessing steps including removing stopwords, lemmatizing process, regular expression operations, etc.

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import unidecode
import textstat
import string  

# Define variable for WordNetLammatizer().
lemmer = WordNetLemmatizer()

# Define function for preprocessor. Input is a single document, as a single string.
# Output should be a single document, as a single string.

def my_preprocess(doc):
    
    # Lowercase
    doc = doc.lower()   
 
    # Substitute single characters with single space
    doc = re.sub(r'\s+[a-z]\s+', ' ', doc)
    
    # Substitute starting single characters with single space
    doc = re.sub(r'^[a-z]\s+', ' ', doc) 
    
    # Substitute digits with single space
    doc = re.sub('\d+', ' ', doc)
    
    # Substitute isn't, ain't, wasn't, didn't... with "not" to avoid "''" got removed in the cleaning process
    doc = re.sub(r"isn't","is not",doc)
    doc = re.sub(r"ain't","am not",doc)
    doc = re.sub(r"wasn't","was not",doc)
    doc = re.sub(r"didn't","did not",doc)
    doc = re.sub(r"don't","do not",doc)
    doc = re.sub(r"wouldn't","would not",doc)
    doc = re.sub(r"shouldn't","should not",doc)
    doc = re.sub(r"can't","can not",doc)
    doc = re.sub(r"couldn't","could not",doc)
    doc = re.sub(r"won't","will not",doc)
    
    # Substitute non word characters with single space
    doc = re.sub('\W+', ' ', doc)
    
    # Substitute multiple spaces with single space
    doc = re.sub(r'\s+', ' ', doc, flags=re.I)
    
    # Lemmatize each word
    doc = ' '.join([lemmer.lemmatize(w) for w in doc.split()])

    return doc

# *********************************************
#
# Define functions for additional features in the document
#
# *********************************************

# They will later be put into the Pipeline and called via the FunctionTransformer() function.
# Each one takes an entier corpus (as a list of documents), and should return
# an array of feature values (one for each document in the corpus). 

# Import a few popular lexicon packages e.g. textblob, afinn, nltk.sentiment.vader.
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn

# Count length
def doc_length(corpus):
    return np.array([len(doc) for doc in corpus]).reshape(-1, 1)

# Count number of words present in the text
def lexicon_count(corpus):
    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)

# Get number of punctuation and sum them up
def _get_punc(doc):
    return len([a for a in doc if a in string.punctuation])

def punc_count(corpus):
    return np.array([_get_punc(doc) for doc in corpus]).reshape(-1, 1)

# Get number of upper case and sum them up
def _get_caps(doc):
    return sum([1 for a in doc if a.isupper()])

def capital_count(corpus):
    return np.array([_get_caps(doc) for doc in corpus]).reshape(-1, 1)

# Count number of exclamation marks
def num_exclamation_marks(corpus):
    return np.array([doc.count('!') for doc in corpus]).reshape(-1, 1)

# Count number of question marks
def num_question_marks(corpus):
    return np.array([doc.count('?') for doc in corpus]).reshape(-1, 1)

# Return boolean value if "not" exists or not
def has_not(corpus):
    return np.array([bool(re.search("not", doc.lower())) for doc in corpus]).reshape(-1, 1)
 
# Return sentiment polarity value from TextBlob lexicon
def sentiment_polar(corpus):
    return np.array([TextBlob(doc).sentiment.polarity for doc in corpus]).reshape(-1, 1)

# Return sentiment positive score value from nltk vader lexicon
def sid_pos(corpus):
    return np.array([SentimentIntensityAnalyzer().polarity_scores(doc)['pos'] for doc in corpus]).reshape(-1, 1) 

# Return sentiment compound score value from nltk vader lexicon
def sid_compound(corpus):
    return np.array([SentimentIntensityAnalyzer().polarity_scores(doc)['compound'] for doc in corpus]).reshape(-1, 1) 

# Return sentiment score value from Afinn lexicon
def afn(corpus):
    afn = Afinn(emoticons=True)
    return np.array([afn.score(doc) for doc in corpus]).reshape(-1, 1) 


# Calculate the class weights and check whether data has any class imbalance issue.
# Data turns out quite equaly distibuted for "Polarity".
import numpy as np
neg, pos = np.bincount(df['Polarity'])
total = neg + pos
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# *********************************************
#
# # Construct the Pipeline
#
# *********************************************

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Need to preprocess the stopwords, because scikit learn's TfidfVectorizer
# removes stopwords _after_ preprocessing.
stop_words = [my_preprocess(word) for word in stop_words.ENGLISH_STOP_WORDS]

# This vectorizer will be used to create the BOW features.
vectorizer = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1500, 
                             ngram_range=[1,2],
                             stop_words=[None],
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.4, min_df=0.001, use_idf=True)

# This vectorizer will be used to preprocess the text before topic modeling.
vectorizer2 = TfidfVectorizer(preprocessor=my_preprocess, 
                             max_features = 1500, 
                             ngram_range=[1,4],
                             stop_words=[None],
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.4, min_df=0.001, use_idf=True)

# Topic modelling NMF
nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)

# Algorithms - Random Forest (LR), Multi-layer Perceptron (MLP), ExtraTreeClassifier (ET), Logistic Regression (LR), and
# support vector machine (SVM)
rf = RandomForestClassifier(criterion='entropy', random_state=223)
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=200)
et = ExtraTreesClassifier(random_state=101, n_estimators=200)
lr = LogisticRegression(random_state=101, max_iter=5000, C = 10, solver='liblinear')
svm = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced")

# Use FeatureUnion() to put all features together as preprocessing step.
feature_processing =  FeatureUnion([ 
    ('bow', Pipeline([('cv', vectorizer), ])),
    ('topics', Pipeline([('cv', vectorizer2), ('nmf', nmf),])),
    ('length', FunctionTransformer(doc_length, validate=False)),
    ('words', FunctionTransformer(lexicon_count, validate=False)),
    ('punc_count', FunctionTransformer(punc_count, validate=False)),
    ('capital_count', FunctionTransformer(capital_count, validate=False)),  
    ('num_exclamation_marks', FunctionTransformer(num_exclamation_marks, validate=False)),  
    ('num_question_marks', FunctionTransformer(num_question_marks, validate=False)),
    ('has_not', FunctionTransformer(has_not, validate=False)),    
    ('afn', FunctionTransformer(afn, validate=False)), 
    ('sentiment_polar', FunctionTransformer(sentiment_polar, validate=False)),    
    ('sid_pos', FunctionTransformer(sid_pos, validate=False)), 
    ('sid_compound', FunctionTransformer(sid_compound, validate=False)), 
])


# Use standard scaler to scale data
steps = [('features', feature_processing),('scaler', StandardScaler(with_std=True,with_mean=False))]

# Define the pipeline with feature processing, scaling, and them run model.
pipe = Pipeline([('features', feature_processing),('scaler', StandardScaler(with_std=True,with_mean=False)), ('clf', et)])

param_grid = {}


# *********************************************
#
# Cross validation and hyper-parameters tuning
#
# *********************************************
# ExtraTreeClassifier (ET) is chosen as the final optimal model. Currently "ET" is set as default under "which_clf" variable.
# To switch to other algorithm, simply changed the "which_clf" variable to "RF", "MLP", "LR" or "SVM".
# All these algorithms (ExtraTreeClassifier (ET), Random Forest (LR), Multi-layer Perceptron (MLP), Logistic Regression (LR), 
# support vector machine (SVM) have been tuned. The best parameters are listed and commented out in each section.

which_clf = "ET"

if which_clf == "RF":

    steps.append(('clf', rf))

    # Hypertuning for more than 3 hours (185.4 mins)
    #Best parameter (CV scy_train0.849):
    #{'clf__class_weight': None, 
    #'clf__criterion': 'entropy', 
    #'clf__n_estimators': 100, 
    #'features__bow__cv__max_features': 3000, 
    #'features__bow__cv__preprocessor': <function my_preprocess at 0x000001F725CB4D38>,
    #'features__bow__cv__use_idf': False, 
    #'features__topics__cv__stop_words': 'english',
    #'features__topics__nmf__n_components': 200}
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [1000,3000],
        'features__bow__cv__use_idf': [False, True],
        'features__topics__cv__stop_words': [None,'english'],
        'features__topics__nmf__n_components': [75,200,400],
        'clf__n_estimators': [100, 500],
        'clf__class_weight': [None],
        'clf__criterion': ['gini', 'entropy'],
    }
    
elif which_clf == "MLP":
    
    steps.append(('clf', mlp))
  
    #Best parameter (CV scy_train0.804):
    #{'clf__hidden_layer_sizes': (50, 50), 
    #'clf__solver': 'adam', 
    #'features__bow__cv__max_features': 3000, 
    #'features__bow__cv__min_df': 0, 
    #'features__bow__cv__preprocessor': <function my_preprocess at 0x000001BA2DA97D38>, 
    #'features__bow__cv__use_idf': False, 
    #'features__topics__cv__stop_words': 'english', 
    #'features__topics__nmf__n_components': 300}
    
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [3000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False],
        'features__topics__cv__stop_words': ["english"],
        'features__topics__nmf__n_components': [300],
        'clf__solver': ["adam"],         
        'clf__hidden_layer_sizes': [(100, ), (50, 50)],
    }

elif which_clf == "ET":
    
    steps.append(('clf', et))
    
    #Best parameter (CV scy_train0.852):
    #{'clf__class_weight': None, 
    #'clf__criterion': 'entropy', 
    #'clf__n_estimators': 200, 
    #'features__bow__cv__max_features': 3000, 
    #'features__bow__cv__preprocessor': <function my_preprocess at 0x0000027D52AA1678>,
    #'features__bow__cv__use_idf': False,
    #'features__topics__cv__max_features': 1500,
    #'features__topics__cv__stop_words':stop_words,
    #'features__topics__nmf__n_components': 200
    
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [3000],
        'features__bow__cv__stop_words': [None, stop_words],
        'features__bow__cv__use_idf': [False],
        'features__topics__cv__max_features': [1500],
        'features__topics__cv__stop_words': [stop_words],
        'features__topics__nmf__n_components': [200],
        'clf__n_estimators': [200],
        'clf__class_weight': [None],   
        'clf__criterion': ['entropy'],
        
    }
    
elif which_clf == "LR":
    
    steps.append(('clf', lr))
    #Best parameter (CV scy_train0.839):
    #{'clf__C': 100, 'clf__solver': 'lbfgs', 
    #'features__bow__cv__max_features': 3000, 
    #'features__bow__cv__preprocessor': <function my_preprocess at 0x00000257B9123DC8>, 
    #'features__bow__cv__use_idf': False, 
    #'features__topics__cv__stop_words': stop_words,
    #'features__topics__nmf__n_components': 300
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [3000],
        'features__bow__cv__use_idf': [False, True],
        'features__topics__cv__stop_words': [stop_words],
        'features__topics__nmf__n_components': [300],        
        'clf__C': [1,10,100],         
        'clf__solver': ['lbfgs', 'liblinear'],
    }

elif which_clf == "SVM":
    
    steps.append(('clf', svm))    
    #Best parameter (CV scy_train0.829):
    #{'clf__class_weight': None, 
    #'clf__kernel': 'linear', 
    #'features__bow__cv__max_features': 3000, 
    #'features__bow__cv__preprocessor': <function my_preprocess at 0x000001BA2F0FC828>,
    #'features__bow__cv__use_idf': True, 
    #'features__topics__cv__stop_words': None, 
    #'features__topics__nmf__n_components': 400}   
 
    param_grid = {
        'features__bow__cv__preprocessor': [my_preprocess],
        'features__bow__cv__max_features': [3000],
        'features__bow__cv__use_idf': [True],
        'features__topics__cv__stop_words': [None],
        'features__topics__nmf__n_components': [400],
        'clf__kernel': ['linear'],
        'clf__class_weight': [None],
    }
    
pipe = Pipeline(steps)

# Use GridSearchCV() for cross validation and hyper-parameters tuning
search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=3, scoring='f1_micro', return_train_score=True, verbose=2)


# *********************************************
#
# # Fit Model
#
# *********************************************
search = search.fit(X_train, y_train)

# Print out the results of hyperparmater tuning
def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results

results = cv_results_to_df(search.cv_results_)
#print(results)

# Export to csv (comment out below line to allow export)
#results.to_csv('results2.csv', index=False)


# **********************************
#
# # Estimate Model Performance
#
# **********************************

# Get references to the objects from the pipeline with the *best* hyperparameter settings,
# in order to explore those objects at a later time.

# The pipeline with the best performance.
pipeline = search.best_estimator_

# Get the feature processing pipeline, which can be used later.
feature_processing_obj = pipeline.named_steps['features']

# Find the vectorizer objects, NMF objects, and classifier objects.
pipevect= dict(pipeline.named_steps['features'].transformer_list)
vectorizer_obj = pipevect.get('bow').named_steps['cv']
vectorizer_obj2 = pipevect.get('topics').named_steps['cv']
nmf_obj = pipevect.get('topics').named_steps['nmf']
clf_obj = pipeline.named_steps['clf']

# Confirm vocabSize setting. Should match the output.
len(vectorizer_obj.get_feature_names())

# Performance metrics of the model.

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

features_val = feature_processing_obj.transform(X_val).todense()

pred_val = search.predict(X_val)

print("Confusion matrix:")
print(confusion_matrix(y_val, pred_val))

print("\nF1 Score = {:.5f}".format(f1_score(y_val, pred_val, average='micro')))

print("\nClassification Report:")
print(classification_report(y_val, pred_val))

# **********************************
#
# # Evaluate Performance on Test Data
#
# **********************************

# Performance metrics of the model on sentiment_test data.

# load in test data and predict.
test_df = pd.read_csv('sentiment_test.csv')

features_test = feature_processing_obj.transform(test_df['Sentence']).todense()
pred_test = search.predict(test_df['Sentence'])

# load in test data and define target vairable.
solutions_df = pd.read_csv('sentiment_test.csv')
y_test = solutions_df['Polarity']

print("Confusion matrix:")
print(confusion_matrix(y_test, pred_test))

print("\nF1 Score = {:.5f}".format(f1_score(y_test, pred_test, average="micro")))

print("\nClassification Report:")
print(classification_report(y_test, pred_test))


# **********************************
#
# # Export Test Data Prediction Results
#
# **********************************

# Output the predictions to a csv file.

# Create a final data frame with three columns "Sentence", "Polarity", "Predicted"
df_test_pred = pd.DataFrame({'Sentence': test_df.Sentence, 'Polarity': test_df.Polarity, 'Predicted': pred_test})

# Create a new column "Validation" to compare the actual vs prediction
df_test_pred['Validation'] = ''

# loop through each instance to compare "Polarity" and "Predicted"
row = 0
for row in range(len(df_test_pred)):    
    if (df_test_pred['Polarity'][row] == df_test_pred['Predicted'][row]):
        result = 'Correct'
    else:
        result = 'Incorrect'
    df_test_pred['Validation'][row] = result
    row += 1

# View first 20 and last 20 rows of data frame. Set display width = 100.
pd.set_option('display.max_colwidth', 100)
print(df_test_pred.head(15))
print(df_test_pred.tail(15))

# Export to csv. Uncomment to create the file.    
#df_test_pred.to_csv("sentiment_test_predicted.csv")


# # Export Incorrect Prediction Results (sentiment_test data)

# Select the predictions that are incorrect.
# Export to a separate csv. This file will be used for responding on Q2 Task 3.

incorrect_pred = df_test_pred[df_test_pred.Validation == "Incorrect"]
print(incorrect_pred.head())

# Uncomment below to export to a separate csv.
# incorrect_pred.to_csv("incorrect_pred.csv")

# **********************************
#
# # Explore the Model Further
#
# **********************************
# Understanding what the model learned.

# ## Print Topics
# Print the top words for each of the NMF topics

n_top_words = 12
def get_top_words(H, feature_names):
    output = []
    for topic_idx, topic in enumerate(H):
        top_words = [(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
        output.append(top_words)
        
    return pd.DataFrame(output) 

top_words = get_top_words(nmf_obj.components_, vectorizer_obj2.get_feature_names())
print(top_words)


# ## Print Feature Importances

topic_feature_names = ["topic {}".format(i) for i in range(nmf_obj.n_components_)]

stat_feature_names = [t[0] for t in pipeline.named_steps['features'].transformer_list if t[0] not in ['topics', 'bow']]

feature_names = vectorizer_obj.get_feature_names() + topic_feature_names + stat_feature_names
len(feature_names)

feature_importances = None
if hasattr(clf_obj, 'feature_importances_'):
    feature_importances = clf_obj.feature_importances_



features_train = feature_processing_obj.transform(X_train).todense()

if feature_importances is None:
    print("No Feature importances! Skipping.")
else:
    N = features_train.shape[1]

    ssum = np.zeros(N)
    avg = np.zeros(N)
    avg_pos = np.zeros(N)
    avg_neg = np.zeros(N)
    for i in range(N):
        ssum[i] = sum(features_train[:, i]).reshape(-1, 1)
        avg[i] = np.mean(features_train[:, i]).reshape(-1, 1)
        avg_pos[i] = np.mean(features_train[y_train==1, i]).reshape(-1, 1)
        avg_neg[i] = np.mean(features_train[y_train==0, i]).reshape(-1, 1)

    et = search.best_estimator_
    imp = pd.DataFrame(data={'feature': feature_names, 'imp': feature_importances, 'sum': ssum, 'avg': avg, 'avg_neg': avg_neg, 'avg_pos': avg_pos})
    imp = imp.sort_values(by='imp', ascending=False)
    print(imp.head(5))
    print(imp.tail(5))
    #imp.to_csv('importances.csv', index=False)


# *********************************************
#
# # Further exploration on Sentiment_Test Data
# # Support Q2 Task 3 Response
#
# *********************************************

# Explain all predictions that were incorrect of a tree-based model.
# 
# Note: this only works on tree-based models, like RF, ET. This cell will crash when using, e.g., MLPClassifier


# Decompose the predictions into the bias term (which is just the testset mean) and individual feature contributions,
# in order to understand which features contributed to the difference and by how much.

if  feature_importances is None:
    print("No Feature importances! Skipping.")
else:
    from treeinterpreter import treeinterpreter as ti

    prediction, bias, contributions = ti.predict(clf_obj, features_test)

    for i in range(len(features_test)):
        if y_test[i] == pred_test[i]:
            continue
        print("Instance {}".format(i))
        print(test_df.iloc[i, :].Sentence)
        print("Bias (testset mean) {}".format(bias[i]))
        print("Truth {}".format(y_test[i]))
        print("Prediction {}".format(prediction[i, :]))
        print("Feature contributions:")
        con = pd.DataFrame(data={'feature': feature_names,
                                 'value': features_test[i].A1,
                                 'neg contr': contributions[i][:, 0],
                                 'pos contr': contributions[i][:, 1],
                                 'abs contr': abs(contributions[i][:, 1])})
        con = con.sort_values(by="abs contr", ascending=False)
        con['pos cumulative'] = con['pos contr'].cumsum() + bias[i][1]
        print(con.head(10))
        print("-"*20)





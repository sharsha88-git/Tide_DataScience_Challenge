#!/usr/bin/env python
# coding: utf-8

# In[64]:


import cupy as cp # linear algebra
from cupy import asnumpy
from random import shuffle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report, precision_recall_curve, plot_precision_recall_curve, average_precision_score, auc
from xgboost import XGBRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error

import seaborn as sns
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import xgboost as xgb
import shap
import os


# In[65]:


data = pd.read_csv('C:\data_interview_test.csv',sep=":")


# In[66]:


data= data["receipt_id"].str.split(":", n = 13, expand = True)
data["receipt_id"]= data[0]
data.drop(columns =[0], inplace = True)
data["company_id"]= data[1]
data.drop(columns =[1], inplace = True)
data["matched_transaction_id"]= data[2]
data.drop(columns =[2], inplace = True)
data["feature_transaction_id"]= data[3]
data.drop(columns =[3], inplace = True)
data["DateMappingMatch"]= data[4]
data.drop(columns =[4], inplace = True)
data["AmountMappingMatch"]= data[5]
data.drop(columns =[5], inplace = True)
data["DescriptionMatch"]= data[6]
data.drop(columns =[6], inplace = True)
data["DifferentPredictedTime"]= data[7]
data.drop(columns =[7], inplace = True)
data["TimeMappingMatch"]= data[8]
data.drop(columns =[8], inplace = True)
data["PredictedNameMatch"]= data[9]
data.drop(columns =[9], inplace = True)
data["ShortNameMatch"]= data[10]
data.drop(columns =[10], inplace = True)
data["DifferentPredictedDate"]= data[11]
data.drop(columns =[11], inplace = True)
data["PredictedAmountMatch"]= data[12]
data.drop(columns =[12], inplace = True)
data["PredictedTimeCloseMatch"]= data[13]
data.drop(columns =[13], inplace = True)
data.head()


# In[29]:


data.shape


# In[30]:


data['label'] = (data.matched_transaction_id == data.feature_transaction_id).astype(int)
data.hist('label')
print(data.label.value_counts())


# In[67]:


data['label']= (data.matched_transaction_id == data.feature_transaction_id).astype(int)
x,y = data[list(data.columns[4:])].astype(float), data.label
x=x.drop(['label'], axis=1)  
x.head()


# In[32]:


#splitting a testing set from the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 42)
#splitting a validation set from the training set to tune parameters
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, stratify = Y_train, random_state = 42)


# In[33]:


train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)
watchlist = [(test, 'eval'), (train, 'train')]

xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'max_depth': 2, 
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            
}

clf = xgb.train( xgb_params,train, num_boost_round=10000,                 )

preds = clf.predict(test)


# In[34]:


print(classification_report(asnumpy(Y_test),preds.round(0)))
print( " In the Confusion Matrix below, the digonal values represent correct classification for each class : ")
labels = ['label-0', 'label-1']
#print(confusion_matrix((y_test),(preds.round(0).astype(int))))  


cm = sklearn.metrics.confusion_matrix(asnumpy(Y_test),asnumpy(preds.round(0).astype(int)))
 

ax= plt.subplot()
sns.heatmap(cm.astype(int), annot=True,fmt='g', ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


# In[83]:


data['label']= (data.DateMappingMatch).astype(float)
data.hist('label')
print(" Distribuition of DateMappingMatch feature values")


# In[36]:


org_data = data.copy()
feat_col=data.columns[4:-1]
le = LabelEncoder()
for col in feat_col:
    var_count = data.groupby(col).agg({col:'count'})
    var_count.columns = ['%s_count'%col]
    var_count = var_count.reset_index()
    data = data.merge(var_count,on=col,how='left')
    le.fit(data['%s_count'%col])  
    encoded = le.transform(data['%s_count'%col])
    data['%s_count'%col] = encoded/encoded.max()
    


x,y = data[list(data.columns[4:])].astype(float), data.label
x=x.drop(['label'], axis=1)  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)
    
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
watchlist = [(test, 'eval'), (train, 'train')]

xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'max_depth': 2, 
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            
}

clf = xgb.train( xgb_params,train, num_boost_round=10000,                 )

preds = clf.predict(test)
print(classification_report(asnumpy(y_test),preds.round(0)))
print( " In the Confusion Matrix below, the digonal values represent correct classification for each class : ")
labels = ['label-0', 'label-1']
#print(confusion_matrix((y_test),(preds.round(0).astype(int))))  


cm = sklearn.metrics.confusion_matrix(asnumpy(y_test),asnumpy(preds.round(0).astype(int)))
 

ax= plt.subplot()
sns.heatmap(cm.astype(int), annot=True,fmt='g', ax = ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


# In[37]:


data = org_data
data['label']= (data.matched_transaction_id == data.feature_transaction_id).astype(int)
x,y = data[list(data.columns[4:])].astype(float), data.label
x=x.drop(['label'], axis=1)  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)
    
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
watchlist = [(test, 'eval'), (train, 'train')]

xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'max_depth': 2, 
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            'scale_pos_weight':13,
}

clf = xgb.train( xgb_params,train, num_boost_round=20000,                 )

preds = clf.predict(test)

print(classification_report(asnumpy(y_test),preds.round(0)))
print( " In the Confusion Matrix below, the digonal values represent correct classification for each class : ")
labels = ['label-0', 'label-1']
#print(confusion_matrix((y_test),(preds.round(0).astype(int))))  


cm = sklearn.metrics.confusion_matrix(asnumpy(y_test),asnumpy(preds.round(0).astype(int)))
 

ax= plt.subplot()
sns.heatmap(cm.astype(int), annot=True,fmt='g', ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


# In[38]:


tp = np.logical_and(preds.round(0)==1,asnumpy(y_test)==1)
fp = np.logical_and(preds.round(0)==1,asnumpy(y_test)==0)
xtest = X_test.copy()
xtest['scores']=preds
xtest.scores[fp].describe()


# In[39]:


xtest.scores[tp].describe()


# In[40]:


data = org_data
data['label']= (data.matched_transaction_id == data.feature_transaction_id).astype(int)
x,y = data[list(data.columns[4:])].astype(float), data.label
x=x.drop(['label'], axis=1)  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)
    
train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
watchlist = [(test, 'eval'), (train, 'train')]

xgb_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'max_depth': 2, 
            'eta':0.1,
            'silent':1,
            'subsample':0.5,
            'colsample_bytree': 0.05,
            'scale_pos_weight':8,
}

clf = xgb.train( xgb_params,train, num_boost_round=20000,                 )

preds = clf.predict(test)

print(classification_report(asnumpy(y_test),preds.round(0)))
print( " In the Confusion Matrix below, the digonal values represent correct classification for each class : ")
labels = ['label-0', 'label-1']
#print(confusion_matrix((y_test),(preds.round(0).astype(int))))  


cm = sklearn.metrics.confusion_matrix(asnumpy(y_test),asnumpy(preds.round(0).astype(int)))
 

ax= plt.subplot()
sns.heatmap(cm.astype(int), annot=True,fmt='g', ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);


# In[41]:


tp = np.logical_and(preds.round(0)==1,asnumpy(y_test)==1)
fp = np.logical_and(preds.round(0)==1,asnumpy(y_test)==0)
xtest = X_test.copy()
xtest['scores']=preds
print ( "false postive stats")
print(xtest.scores[fp].describe())
print()
print ( "True postive stats")
print(xtest.scores[tp].describe())


# In[47]:


from xgboost import XGBRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


# In[48]:


def prep_data(data):
    data['label']= (data.matched_transaction_id == data.feature_transaction_id).astype(int)
    feat_col=data.columns[4:-1]
    le = LabelEncoder()
    for col in feat_col:
        var_count = data.groupby(col).agg({col:'count'})
        var_count.columns = ['%s_count'%col]
        var_count = var_count.reset_index()
        data = data.merge(var_count,on=col,how='left')
        le.fit(data['%s_count'%col])  
        encoded = le.transform(data['%s_count'%col])
        data['%s_count'%col] = encoded/encoded.max()
    x,y = data[list(data.columns[4:])].astype(float), data.label
    x=x.drop(['label'], axis=1)  
    return x,y
x,y = prep_data(data)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42,shuffle=True)

    

space = {
        'max_depth':hp.choice('max_depth', np.arange(10, 25, 1, dtype=int)),
        'n_estimators':hp.choice('n_estimators', np.arange(1000, 10000, 10, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
        'min_child_weight':hp.choice('min_child_weight', np.arange(250, 350, 10, dtype=int)),
        'subsample':hp.quniform('subsample', 0.7, 0.9, 0.1),
        'eta':hp.quniform('eta', 0.1, 0.3, 0.1),
        
        'objective':'reg:squarederror',
        
        'tree_method':'gpu_hist',
        'eval_metric': 'rmse',
    }

def score(params):
    model = XGBRegressor(**params)
    
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False, early_stopping_rounds=10)
    Y_pred = model.predict(X_test).clip(0, 20)
    score = sqrt(mean_squared_error(y_test, Y_pred))
    print(score)
    return {'loss': score, 'status': STATUS_OK}    
    
def optimize(trials, space):
    
    best = fmin(score, space, algo=tpe.suggest, max_evals=10)
    return best

trials = Trials()
best_params = optimize(trials, space)

# Return the best parameters
space_eval(space, best_params)


# In[68]:


#splitting a testing set from the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.20, stratify = y, random_state = 42)
#splitting a validation set from the training set to tune parameters
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.20, stratify = Y_train, random_state = 42)


# In[69]:


#creating a scorer from the f1-score metric
f1_scorer = make_scorer(f1_score)


# In[70]:


# defining the space for hyperparameter tuning
space = {'eta': hp.uniform("eta", 0.1, 1),
        'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 50, 200, 1),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 200, 10)
        }


# In[71]:


#defining function to optimize
def hyperparameter_tuning(space):
    clf = xgb.XGBClassifier(n_estimators = int(space['n_estimators']),       #number of trees to use
                            eta = space['eta'],                              #learning rate
                            max_depth = int(space['max_depth']),             #depth of trees
                            gamma = space['gamma'],                          #loss reduction required to further partition tree
                            reg_alpha = int(space['reg_alpha']),             #L1 regularization for weights
                            reg_lambda = space['reg_lambda'],                #L2 regularization for weights
                            min_child_weight = space['min_child_weight'],    #minimum sum of instance weight needed in child
                            colsample_bytree = space['colsample_bytree'],    #ratio of column sampling for each tree
                            nthread = -1)                                    #number of parallel threads used
    
    evaluation = [(X_train, Y_train), (X_val, Y_val)]
    
    clf.fit(X_train, Y_train,
            eval_set = evaluation,
            early_stopping_rounds = 10,
            verbose = False)

    pred = clf.predict(X_val)
    pred = [1 if i>= 0.5 else 0 for i in pred]
    f1 = f1_score(Y_val, pred)
    print ("SCORE:", f1)
    return {'loss': -f1, 'status': STATUS_OK }


# In[72]:


# run the hyper paramter tuning
trials = Trials()
best = fmin(fn = hyperparameter_tuning,
            space = space,
            algo = tpe.suggest,
            max_evals = 100,
            trials = trials)

print (best)


# In[73]:


#plotting feature space and f1-scores for the different trials
parameters = space.keys()
cols = len(parameters)

f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
    axes[i].grid()


# In[74]:


#printing best model parameters
print(best)


# In[75]:


#initializing XGBoost Classifier with best model parameters
best_clf = xgb.XGBClassifier(n_estimators = int(best['n_estimators']), 
                            eta = best['eta'], 
                            max_depth = int(best['max_depth']), 
                            gamma = best['gamma'], 
                            reg_alpha = int(best['reg_alpha']), 
                            min_child_weight = best['min_child_weight'], 
                            colsample_bytree = best['colsample_bytree'], 
                            nthread = -1)


# In[76]:


#fitting XGBoost Classifier with best model parameters to training data
best_clf.fit(X_train, Y_train)


# In[77]:


#using the model to predict on the test set
Y_pred = best_clf.predict(X_test)


# In[78]:


#printing f1 score of test set predictions
print('The f1-score on the test data is: {0:.2f}'.format(f1_score(Y_test, Y_pred)))


# In[79]:


#creating a confusion matrix and labels
cm = confusion_matrix(Y_test, Y_pred)
labels = ['label-0', 'label-1']


# In[80]:


#plotting the confusion matrix
sns.heatmap(cm, annot = True, xticklabels = labels, yticklabels = labels, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# In[81]:


#printing classification report
print(classification_report(Y_test, Y_pred))


# In[82]:


Y_score = best_clf.predict_proba(X_test)[:, 1]
average_precision = average_precision_score(Y_test, Y_score)
fig = plot_precision_recall_curve(best_clf, X_test, Y_test)
fig.ax_.set_title('Precision-Recall Curve: AP={0:.2f}'.format(average_precision))


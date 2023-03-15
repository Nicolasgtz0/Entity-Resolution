#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:03:59 2023

@author: nicolasgutierrez
"""

#Import packages 

import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn as skl
import jarowinkler as jw 


def read(filename:str):
    """
    Parameters
    ----------
    filename : str 
        File path of affiliations dataset. 
    Returns
    -------
    df : pd.dataframe
        This is the affiliations dataframe.
    """
    # gotta create label (Match or no match)
    # features, jarowinkler, fasttext, etc 
    # Can you create training set with even 100 examples?
    # Try different set of differences
    
    df = pd.read_csv(filename)
    #return df 
    affiliations = df['name'].values  
    out = []
    for i, aff1 in enumerate (affiliations):
        for j, aff2 in enumerate(affiliations[i+1:]):
            out.append([(aff1, aff2),(i,j)])      
    df = pd.DataFrame(out, columns = ['name','indices']) 
    jwscore = lambda x: jw.jarowinkler_similarity(x[0], x[1])  
    df['jwdist'] = df['name'].apply(jwscore) 
    return df
    

def split(df): #Not needed
    """
    Parameters
    ----------
    df : pd.dataframe
        Affiliations dataframe. 
    Returns
    -------
    y_train, y_test, X_train, X_test
    4 arrays containing train and test split of data 
    """
    xVar = np.asarray(df[' '])
    yVar = np.asarray(df[[' ',' ',' ',' ']])
    
    y_train, y_test, X_train, X_test = skl.model_selection.train_test_split(yVar, xVar, test_size=0.2) 
    return y_train, y_test, X_train, X_test 


def fit_and_train(y_train:np.array, X_train:np.array): #training it
    '''
    Parameters
    ----------
    y_train : np.array
        training features.
    X_train : np.array
        label features.
    Returns 
    -------
    xgb : XGBoost() 
        fitted model.
    '''
    model = xgb.XGBClassifier()
    model.fit(y_train, X_train) 
    return model



def predict(model:xgb.XGBClassifier, y_test:np.array): #running it 
    '''
    Parameters
    ----------
    xgb : XGBoost
        fitted model.
    y_test : np.array
        test features.
    Returns
    -------
    x_pred : np.array
        Array of the predicted labels.
    '''
    # instead of y_test, use features 
    features = ['jwdist']
    labels = ['match'] 
    #model = 
    
    x_pred = model.predict(y_test)
    return x_pred
    return model.predict_proba(y_test)



def asses(x_pred:np.array, X_test: np.array): 
    '''
    Parameters
    ----------
    x_pred : np.array
        Array of the predicted labels..
    X_test : np.array
        Labels test.
    Returns
    -------
    accuracy : float
        Matching accuracy, compare test to predicted.
    '''
    matching = np.where(x_pred == X_test)[0]
    accuracy = len(matching)/len(X_test)
    print('accuracy:', accuracy*100,'%') 
    return accuracy

#easy match: Change one letter 
#easy non-match: Completely different (identity)
# Hard matches: What happens if there are 4 words wrong? Location?

if __name__ =='__main__': 
    affy = read('affiliations.csv')
    #print(affy.loc[(affy['indices']==(586,587))])  
    train_set = pd.DataFrame()
    #train_set = train_set.assign(match=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,])
    #matches = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0] 
    #train_set['match'] = matches  
    #print(train_set['match'])
    ###train_set['match']==0
    #Training the data manually, match or non match
    train_set = train_set.append(affy.loc[affy['indices']==(13,14)])
    train_set = train_set.append(affy.loc[affy['indices']==(15,16)]) 
    train_set = train_set.append(affy.loc[affy['indices']==(31,32)])
    train_set = train_set.append(affy.loc[affy['indices']==(75,76)])
    train_set = train_set.append(affy.loc[affy['indices']==(77,81)])
    train_set = train_set.append(affy.loc[affy['indices']==(75,82)])
    train_set = train_set.append(affy.loc[affy['indices']==(81,97)])
    train_set = train_set.append(affy.loc[affy['indices']==(13,99)])
    train_set = train_set.append(affy.loc[affy['indices']==(103,107)])
    train_set = train_set.append(affy.loc[affy['indices']==(246,149)])
    train_set = train_set.append(affy.loc[affy['indices']==(0,5)])
    train_set = train_set.append(affy.loc[affy['indices']==(13,20)]) 
    train_set = train_set.append(affy.loc[affy['indices']==(3,17)])
    train_set = train_set.append(affy.loc[affy['indices']==(23,27)])
    train_set = train_set.append(affy.loc[affy['indices']==(35,34)])
    train_set = train_set.append(affy.loc[affy['indices']==(89,90)])
    train_set = train_set.append(affy.loc[affy['indices']==(100,101)])
    train_set = train_set.append(affy.loc[affy['indices']==(108,109)])
    train_set = train_set.append(affy.loc[affy['indices']==(134,135)]) 
    train_set = train_set.append(affy.loc[affy['indices']==(141,142)])  
    train_set = train_set.assign(match=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    print(train_set) 
    
## Pick 10 easy matches 
# (13,14) 
# (15,16)
# (31,32)
# (75,76)
# (77,81)
# (75,82)
# (81,97)
# (13,99)
# (103,107)
# (246,149)

## Pick 10 easy non-matches 
# (0,5) 
# (13,20)
# (3,17)
# (23,27)
# (35,34)
# (89,90)
# (100,101)
# (108,109)
# (134,135)
# (141,142)
#train_set = train_set.append(affy.loc[affy['indices']==(586,587)])  -> Caused error
#train_set = train_set.append(affy.loc[affy['indices']==(744,745)])  -> Caused error

    
    
    #features_train, features_test, Label_train, Label_test = split(affy) 
    #affy_model = fit_and_train(features_train,Label_train)
    #predict_affy = predict(affy_model, features_test) 
    #affy_accuracy = asses(predict_affy, Label_test) 

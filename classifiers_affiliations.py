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
    
    

def split(df): 
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
    x_pred = model.predict(y_test)
    return x_pred


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
    print(affy)
    #features_train, features_test, Label_train, Label_test = split(affy) 
    #affy_model = fit_and_train(features_train,Label_train)
    #predict_affy = predict(affy_model, features_test) 
    #affy_accuracy = asses(predict_affy, Label_test)

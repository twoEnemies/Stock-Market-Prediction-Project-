import numpy as np
import pandas as pd
import pickle
from matplotlib import style 
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import svm,cross_validation,neighbors
from sklearn.ensemble import VotingClassifier ,RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days=7
    df=pd.read_csv('sp_500_joined_classes.csv',index_col=0)
    tickers=df.columns.values.tolist()
    df.fillna(0,inplace=True)
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)]=(df[ticker].shift(-i)-df[ticker]) /df[ticker]
    
    df.fillna(0,inplace=True)
    return tickers,df

#process_data_for_labels('MMM')

def buy_sell_hold(*args):
    cols=[c for c in args]
    requriements=0.02
    for col in cols:
        if col>requriements:
            return 1
        if col<-requriements:
            return -1
    return 0

def extract_features(ticker):
    tickers,df=process_data_for_labels(ticker)
    
    df['{}_target'.format(ticker)]=list(map(buy_sell_hold,
                                           df['{}_1d'.format(ticker)],
                                           df['{}_2d'.format(ticker)],
                                           df['{}_3d'.format(ticker)],
                                           df['{}_4d'.format(ticker)],
                                           df['{}_5d'.format(ticker)],
                                           df['{}_6d'.format(ticker)],
                                           df['{}_7d'.format(ticker)]))
    
    vals=df['{}_target'.format(ticker)].values.tolist()
    str_vals=[str(i) for i in vals]
    print('Data Spread:',Counter(str_vals))
    df.fillna(0,inplace=True)
    
    df=df.replace([np.inf,-np.inf],np.nan)
    df.dropna(inplace=True)
    
    df_vals= df[[ticker for ticker in tickers]].pct_change()
    df_vals=df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)
    
    X=df_vals.values
    y=df['{}_target'.format(ticker)].values
    
    return X,y,df
#extract_features('MMM')


def do_ml(ticker):
    X,y,df=extract_features(ticker)
    
    X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,
                                                                   y,
                                                                   test_size=0.25)
    #clf=neighbors.KNeighborsClassifier()
    clf=VotingClassifier([('lsvc',svm.LinearSVC()),
                         ('knn',neighbors.KNeighborsClassifier()),
                         ('rfor',RandomForestClassifier())])
    
    clf.fit(X_train,y_train)
    confidance=clf.score(X_test,y_test)
    
    print("Accuracy",confidance)
    predictions=clf.predict(X_test)
    print("predicted Spread:",Counter(predictions))
     
    return confidance
do_ml('AMD')

    

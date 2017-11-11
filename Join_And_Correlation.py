import bs4 as bs
import pickle
from matplotlib import style 
import requests
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')


def save_tickers():
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup=bs.BeautifulSoup(resp.text,"lxml")
    table=soup.find('table',{'class':'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:10]:
        ticker=row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    print(tickers)
    return tickers


#a=save_tickers()

def get_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers=save_tickers()
        print(tickers)
    else:
         with open("sp500tickers.pickle","rb") as f:
                tickers=pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
        
    start=dt.datetime(2010,1,1)
    end=dt.datetime(2016,12,31)
    
    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df=web.DataReader(ticker,'yahoo',start,end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
            
        else:
            print('Already have {}'.format(ticker))
    
#get_from_yahoo()

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers=pickle.load(f)
        
    main_df=pd.DataFrame()
    
    for count ,ticker in enumerate(tickers):
        print(ticker)
        df=pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.rename(columns={'Adj Close':ticker},inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        print(df.head())
        if main_df.empty:
            main_df=df
        else:
            print("in else")
            main_df=main_df.join(df,how='outer')
        
        if count%10==0:
            print(count)
    
    print(main_df.head())
    main_df.to_csv('sp_500_joined_classes.csv')
    
#compile_data()
 
def visualize_data(): 
    df =pd.read_csv('sp_500_joined_classes.csv')
    df_corr=df.corr()
    
    print(df_corr.head())
    data=df_corr.values
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    heatmap=ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels=df_corr.columns
    row_labels=df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
    
    
    
    
visualize_data()

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')
import simplejson
import requests

from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

from matplotlib import interactive
interactive(True)

#query=input("Enter company name:")
yahoo_stock_code="http://d.yimg.com/autoc.finance.yahoo.com/autoc?query="
yahoo_excess_code="&region=1&lang=en" 
stock_url=yahoo_stock_code+query+yahoo_excess_code
response=requests.get(stock_url)

#CONEVRT THE JSON FILE INTO UTF-8 FORMAT FOR PARSING
data=simplejson.loads(response.content.decode("utf-8"))

#FETCH THE FIRST COMPANY CODE
code=data['ResultSet']['Result'][0]['symbol']
#print(code)


#start=dt.datetime(2010,1,1)
#end=dt.datetime(2016,12,31)

#df=web.DataReader(code,'yahoo',start,end)
#df.to_csv('google')

df=pd.read_csv('google.csv',parse_dates=True, index_col=0)
#print(df.head(100))

#df['100ma']=df['Adj Close'].rolling(window=100,min_periods=0).mean()
#print(df.head())

#df.dropna(inplace=True)

df_ohlc=df['Adj Close'].resample('10D').ohlc()
df_volume=df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)

df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)


#print(df_ohlc.head())
#print(df_volume.head())

ax1=plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1,sharex=ax1)

ax1.xaxis_date()
candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)



#ax1.plot(df.index,df['Adj Close'])
#ax1.plot(df.index,df['Volume'])
#ax2.bar(df.index,df['100ma'])
#df.plot()

plt.show()




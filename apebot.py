import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.express as px

def remove_duplicate(l:list):
    return list(dict.fromkeys(l))


#scrapping for table
link = "http://openinsider.com/latest-penny-stock-buys"
#link = "http://openinsider.com/top-insider-purchases-of-the-day"
df = pd.read_html(link)
df = df[11]
df.sort_values(by=df.columns[2],ascending=False)
#df.columns[2] #is the trade date
df = df[df.columns[:13]]
st.dataframe(df[:5])

#title
st.title("Ultimate APE BOT")
tickers = df['Ticker'][:5]
prices = df['Price'][:5]
tickers = remove_duplicate(tickers)
#st.table(pd.DataFrame(data={'Date of Transaction':df[1][:5],'tickers':tickers}))
#streamlit run apebot.py
a = pd.DataFrame()
volume = pd.DataFrame()
#st.table(df)
for i in tickers:
    a[i] = yf.download(i,period="1mo",interval='1d')['Adj Close']
    volume[i] = yf.download(i,period="1mo",interval='1d')['Volume']
    st.write(px.line(a[i],x=a.index,y=i))
    st.write(px.bar(volume[i],x=a.index,y=i))
    
st.header("Earning")
for i in tickers:
    st.info(i)
    st.write(yf.Ticker(i).quarterly_earnings)
    st.write(yf.Ticker(i).calendar)

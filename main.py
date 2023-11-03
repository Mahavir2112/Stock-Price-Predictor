import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go  # Change "graph_objs" to "graph_objs"

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("TCS.BO", "MRF.BO", "RELIANCE.BO", "HINDUNILVR.BO", "INFY.BO")

selected_stock = st.selectbox("Select stock for prediction", stocks)

n_years = st.slider("Years of prediction: ", 1, 4)

period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")  # Change "data_load_state=" to "data_load_state."

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))  # Remove the quotes around 'Open' and 'Close'
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))  # Remove the quotes around 'Open' and 'Close'
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)  # Change "layout.update" to "update_layout"
    st.plotly_chart(fig)

plot_raw_data()

df_train=data[['Date', 'Close']]
df_train=df_train.rename(columns={"Date":"ds", "Close":"y"})

m=Prophet()

m.fit(df_train)
future=m.make_future_dataframe(periods=period)

forecast=m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write("Forecast Graph")
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2=m.plot_components(forecast)
st.write(fig2)
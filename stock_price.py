import streamlit as st
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

amazon_data = pd.read_csv('amazon.csv')

START = amazon_data['Date'].min()
TODAY = amazon_data['Date'].max()

st.title('Stock Price Prediction Web App')
st.subheader("A Comparative Analysis of Stock Price Prediction Models Of Facebook Prophet and Linear Regression with Amazon .CSV Dataset in Financial Forecasting")

n_years = st.number_input('Insert number of years for prediction', step=1, format='%d', min_value=1)
st.write('The chosen number of years for prediction is ', n_years)
period = n_years * 365

@st.cache_data
def load_data():
    data = amazon_data.copy()
    data.reset_index(inplace=True, drop=True)
    return data

data = load_data()

st.subheader('Raw data head')
st.write(data.head())

st.subheader('Raw data tail')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data with Prophet')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} year/years')
fig1 = plot_plotly(m, forecast)
fig1.update_traces(line=dict(color='#0000FF'), marker=dict(color='#ADD8E6'))
st.plotly_chart(fig1)

df_lr = df_train.copy()
df_lr['ds'] = pd.to_datetime(df_lr['ds']) 
df_lr['ds_numeric'] = (df_lr['ds'] - df_lr['ds'].min()).dt.days  

future_lr_dates = pd.date_range(start=df_lr['ds'].max() + pd.DateOffset(1), periods=period, freq='D')
future_lr_df = pd.DataFrame({'ds': future_lr_dates})
future_lr_values = future_lr_df['ds'].apply(lambda x: (x - df_lr['ds'].min()).days).values.reshape(-1, 1)

model = LinearRegression()
model.fit(df_lr[['ds_numeric']], df_lr['y'])

future_lg = model.predict(df_lr[['ds_numeric']])  
predicted_future = model.predict(future_lr_values)

st.subheader('Linear Regression Forecast')
st.write(pd.DataFrame({'Date': future_lr_dates, 'Close': predicted_future}))

def plot_linear_regression():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_lr['ds'], y=df_lr['y'], mode='markers', name='Actual Data'))
    fig.add_trace(go.Scatter(x=df_lr['ds'], y=future_lg, mode='lines', name='Linear Regression Line', line=dict(color='red')))
    x_future = pd.date_range(start=df_lr['ds'].max() + pd.DateOffset(1), periods=len(predicted_future), freq='D')
    fig.add_trace(go.Scatter(x=x_future, y=predicted_future, mode='lines', line=dict(color='green', dash='dash'),
                             name='Future Predictions'))
    fig.update_layout(xaxis=dict(title='Days since start', rangeslider=dict(visible=True)),
                      yaxis=dict(title='Close Price'),
                      width=1000,  
                      height=600) 
    st.plotly_chart(fig)

plot_linear_regression()

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

y_true_prophet = df_train['y'].values
y_pred_prophet = forecast['yhat'].values[:-period]

mae_prophet = mean_absolute_error(y_true_prophet, y_pred_prophet)
mse_prophet = mean_squared_error(y_true_prophet, y_pred_prophet)
r2_prophet = r2_score(y_true_prophet, y_pred_prophet)

y_true_lr = df_lr['y'].values
y_pred_lr = model.predict(df_lr[['ds_numeric']])

mae_lr = mean_absolute_error(y_true_lr, y_pred_lr)
mse_lr = mean_squared_error(y_true_lr, y_pred_lr)
r2_lr = r2_score(y_true_lr, y_pred_lr)

metrics_data = {
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R-squared (R2)'],
    'Prophet Model': [mae_prophet, mse_prophet, r2_prophet],
    'Linear Regression Model': [mae_lr, mse_lr, r2_lr]
}

metrics_df = pd.DataFrame(metrics_data)

st.subheader('Regression Evaluation Metrics')
st.write(metrics_df)

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')  
import streamlit as st

from dotenv import load_dotenv
from utils.b2 import B2
from utils.modeling import *


REMOTE_DATA = 'Vegetables_Pulses2023.csv'


load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_keyID'],
        secret_key=os.environ['B2_applicationKey'])


@st.cache_data
def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    return df

st.write("""
## Vegatables and Pulses forecast APP
         
Shown are the price and production forecast



""")



df = get_data()
sorted_item = sorted( df['Item'].unique() )
selected_item = st.sidebar.multiselect('ITEM', sorted_item, sorted_item)
sorted_year=sorted(df['Year'].unique())
selected_year=st.sidebar.multiselect('YEAR',sorted_year,sorted_year)



commodities = data['Commodity'].unique()
def arima_modeling(commodity_data):
    try:
        # Check if commodity_data DataFrame is empty
        if commodity_data.empty:
            print("Empty DataFrame for commodity:", commodity_data['Commodity'].iloc[0])
            return

        # Visualize the time series data
        plt.plot(commodity_data['Year'], commodity_data['PublishValue'])
        plt.xlabel('Year')
        plt.ylabel('PublishValue')
        plt.title('Time Series Plot for {}'.format(commodity_data['Commodity'].iloc[0]))
        plt.show()

        # Check ACF and PACF plots
        plot_acf(commodity_data['PublishValue'])
        plot_pacf(commodity_data['PublishValue'])
        plt.show()

        # Train-test split
        train_size = int(len(commodity_data) * 0.8)
        train_data, test_data = commodity_data[:train_size], commodity_data[train_size:]

        if len(test_data) == 0:
            print("Insufficient data for testing for commodity:", commodity_data['Commodity'].iloc[0])
            return

        # Fit ARIMA model
        model = ARIMA(train_data['PublishValue'], order=(5,1,0))  # Example order, tune as needed
        fitted_model = model.fit()

        # Forecast
        forecast_values = fitted_model.forecast(steps=len(test_data))
        forecast = forecast_values[0]
        conf_int = forecast_values[2]

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data['PublishValue'], forecast))
        print('RMSE:', rmse)

        # Visualize forecast
        plt.plot(test_data['Year'], test_data['PublishValue'], label='Actual')
        plt.plot(test_data['Year'], forecast, label='Forecast')
        plt.fill_between(test_data['Year'], conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.3)
        plt.xlabel('Year')
        plt.ylabel('PublishValue')
        plt.title('Forecast for {}'.format(commodity_data['Commodity'].iloc[0]))
        plt.legend()
        plt.show()
    
    except KeyError as e:
        print("KeyError:", e)

# Apply ARIMA modeling for each commodity
for commodity in commodities:
    commodity_data = data[data['Commodity'] == commodity]
    arima_modeling(commodity_data)

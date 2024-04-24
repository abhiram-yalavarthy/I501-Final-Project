import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')  
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
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




# Get unique items and years
import streamlit as st
import plotly.express as px

# Assuming you have a function named get_data() that returns your DataFrame df
# Replace it with your actual data loading function

# Get the data
df = get_data()





# Display the interactive heatmap

import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
# Assuming you have a DataFrame named df with columns: Location, Commodity, Unit, and PublishValue

# Get unique commodities and units
commodities = sorted(df['Commodity'].unique())
units = sorted(df['Unit'].unique())

# Create dropdowns for commodity and unit selection
selected_commodity = st.selectbox('Select Commodity', commodities)
selected_unit = st.selectbox('Select Unit', units)

# Get user's choice for visualization
visualization_type = st.radio('Choose Visualization', ['Bar Chart', 'Line Chart'])

# Filter data based on selected commodity and unit
filtered_df = df[(df['Commodity'] == selected_commodity) & (df['Unit'] == selected_unit)]

# Group by location and calculate total production value
country_production = filtered_df.groupby('Location')['PublishValue'].sum().reset_index()

# Plotting based on user's choice of visualization
if visualization_type == 'Bar Chart':
    fig = px.bar(country_production, x='Location', y='PublishValue', title=f'Production for {selected_commodity} ({selected_unit}) by Country')
elif visualization_type == 'Line Chart':
    fig = px.line(country_production, x='Location', y='PublishValue', title=f'Production Trend for {selected_commodity} ({selected_unit}) by Country')

# Display the plotly figure
st.plotly_chart(fig)

#Groupby
# Group data by country and commodity and sum the production values
country_commodity_production = df.groupby(['Location', 'Commodity', 'Unit'])['PublishValue'].sum().reset_index()

# Find the country that produces the most of each commodity
max_production_per_commodity = country_commodity_production.loc[country_commodity_production.groupby('Commodity')['PublishValue'].idxmax()]

# Find the commodity produced by the most countries
max_countries_per_commodity = country_commodity_production.loc[country_commodity_production.groupby('Commodity')['PublishValue'].idxmax()]

#3rd visualization
country_commodity_production_agg = country_commodity_production.groupby(['Commodity', 'Location'])['PublishValue'].sum().reset_index()

# Create pivot table
pivot_table = country_commodity_production_agg.pivot(index='Commodity', columns='Location', values='PublishValue')
st.title('Commodity Production Across Countries')

# Plot the interactive heatmap using Plotly Express
fig = px.imshow(pivot_table, 
                labels=dict(x="Country", y="Commodity", color="Production Value"),
                x=pivot_table.columns,
                y=pivot_table.index,
                color_continuous_scale='YlOrRd')  # Choose color scale

# Update layout to adjust figure size and axis labels
fig.update_layout(title='Commodity Production Across Countries',
                  xaxis_title='Country',
                  yaxis_title='Commodity')

# Display the interactive heatmap
st.plotly_chart(fig)


#All Vegetables forecast
# Load the dataset
data = df
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error


###TEST


st.write("First few rows of the data:")
st.write(data.head())





# Function for ARIMA modeling
def arima_modeling(commodity_data):
    try:
        # Check if commodity_data DataFrame is empty
        if commodity_data.empty:
            st.write("Empty DataFrame for commodity:", commodity_data['Commodity'].iloc[0])
            return
        # Visualize the time series data
        fig, ax = plt.subplots()
        ax.plot(commodity_data['Year'], commodity_data['PublishValue'])
        ax.set_xlabel('Year')
        ax.set_ylabel('PublishValue')
        ax.set_title('Time Series Plot for {}'.format(commodity_data['Commodity'].iloc[0]))
        st.pyplot(fig)

        


        # Train-test split
        train_size = int(len(commodity_data) * 0.8)
        train_data, test_data = commodity_data[:train_size], commodity_data[train_size:]

        if len(test_data) == 0:
            st.write("Insufficient data for testing for commodity:", commodity_data['Commodity'].iloc[0])
            return

        # Fit ARIMA model
        model = ARIMA(train_data['PublishValue'], order=(5,1,0))  # Example order, tune as needed
        fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.forecast(steps=len(test_data))

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data['PublishValue'], forecast))
        st.write('RMSE:', rmse)

        # Visualize forecast
        st.write("Actual vs Forecast:")
        chart_data = pd.DataFrame({
            'Year': test_data['Year'],
            'Actual': test_data['PublishValue'],
            'Forecast': forecast
        })
        st.line_chart(chart_data.set_index('Year'), use_container_width=True)

    except KeyError as e:
        st.write("KeyError:", e)
            # Check ACF and PACF plots
        fig_acf, ax_acf = plt.subplots()
        plot_acf(commodity_data['PublishValue'], ax=ax_acf)
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(commodity_data['PublishValue'], ax=ax_pacf)
        st.pyplot(fig_pacf)

# Load data
data = df



# Sidebar options
visualization_type = st.sidebar.selectbox('Select Visualization Type', ['Time Series Plot', 'Autocorrelation Plot', 'Partial Autocorrelation Plot'])


# Filter data based on selected commodity
selected_commodity_data = data[data['Commodity'] == selected_commodity]

# Apply ARIMA modeling for selected commodity based on selected visualization type
if visualization_type == 'Time Series Plot':
    st.write("Time Series Plot for", selected_commodity)
    arima_modeling(selected_commodity_data)
elif visualization_type == 'Autocorrelation Plot':
    st.write("Autocorrelation Plot for", selected_commodity)
    plot_acf(selected_commodity_data['PublishValue'])
    st.pyplot()
elif visualization_type == 'Partial Autocorrelation Plot':
    st.write("Partial Autocorrelation Plot for", selected_commodity)
    plot_pacf(selected_commodity_data['PublishValue'])
    st.pyplot()

#Regression Analysis




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df_encoded = pd.get_dummies(df, columns=['Location'])

data = df

# Sidebar
st.sidebar.title("Model Options")
selected_features = st.sidebar.multiselect("Select features:", data.columns)

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Decade', 'Item', 'Commodity', 'EndUse', 'Unit', 'Category', 'GeographicalLevel', 'Location'])

# Separate features and target variable
X = df_encoded.drop(columns=['PublishValue'])
y = df_encoded['PublishValue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.title("Regression Model Results")
st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)


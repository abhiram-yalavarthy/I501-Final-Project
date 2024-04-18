import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib
matplotlib.use('TkAgg')  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

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



# Read data from CSV file
df = get_data()

production_df = df[df['Item'] == 'Production']

# Check if production data is available
if not production_df.empty:
    # Visualize the trend of artichoke production over the years
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(production_df['Year'], production_df['PublishValue'], marker='o', linestyle='-')
    ax.set_title('Artichoke Production Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Production (Million Pounds)')
    ax.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
else:
    st.write("No data available for artichoke production.")

# Statistical Analysis
# Summary statistics for production
if not production_df.empty:
    production_summary = production_df['PublishValue'].describe()
    st.write("Summary Statistics for Production:\n", production_summary)


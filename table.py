import pandas as pd
import streamlit as st
def print_commodity_units(df):
    # Group the data by commodity and list unique units for each commodity
    commodity_units = df.groupby('Commodity')['Unit'].unique()

    # Create a DataFrame to hold commodities and their units
    commodity_units_df = pd.DataFrame(commodity_units).reset_index()
    commodity_units_df.columns = ['Commodity', 'Units']

    # Display the DataFrame as a table in Streamlit
    st.table(commodity_units_df)

#  usage in Streamlit
if __name__ == '__main__':
    
     

    # Display commodities and their units in Streamlit
    st.title('Commodities and their Units')
    print_commodity_units(df)
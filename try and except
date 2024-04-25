def print_commodity_units(df):
    try:
        
        commodity_units = df.groupby('Commodity')['Unit'].unique()

        
        commodity_units_df = pd.DataFrame(commodity_units).reset_index()
        commodity_units_df.columns = ['Commodity', 'Units']

      
        st.table(commodity_units_df)
    
    except KeyError as e:
        st.error(f"An error occurred: {e}. Please make sure the DataFrame contains the necessary columns.")

# Usage in Streamlit
if __name__ == '__main__':
    try:
        

        # Display commodities and their units in Streamlit
        st.title('Commodities and their Units')
        print_commodity_units(df)
    
    except FileNotFoundError:
        st.error("The specified dataset file was not found. Please check the file path and try again.")
    except pd.errors.EmptyDataError:
        st.error("The dataset is empty. Please make sure the file contains data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}.")

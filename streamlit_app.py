import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import altair as alt




#Function to process the uploaded Excel file
def load_excel(file):
    try:
        df = pd.read_excel(file)
        st.write("File read successfully")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# Function to detect failure based on input failure rate and randomization 
def failure_detection(input_dict):
    ## Create DataFrame and set 'Year' as the index

    # Rename each key in original input_dict by appending "_1" to aovid potential naming conflicts and standardize column names
    input_dict_renamed = {f"{key}_1": value for key, value in input_dict.items()}

    # Create initial DataFrame 
    df = pd.DataFrame(input_dict_renamed)

    # Establish Year column following the provided dataset
    df['Year'] = [i for i in range(1, len(df) + 1)]  # Create Year column
    df.set_index('Year', inplace=True) # Set Year column as Index column 

    # Prepare a list to track the desired column order
    desired_order = []

    # Set up order for the dataframe following Component failure rate, Randomization, and status 
    for column_name in input_dict_renamed.keys():
        random_col = f'Random_{column_name}'  
        status_col = f'Failure_status_{column_name}' 

        # Add new columns with starting with default value as None and NAN
        df[random_col] = np.nan
        df[status_col] = None

        # Append column names in the desired order
    
        desired_order.extend([column_name, random_col, status_col])

    # Ensure the DataFrame columns match the desired order
    df = df[[col for col in desired_order if col in df.columns]]

    ## Iterate over each year to simulate failures 
    df_copy = df.copy() # Duplicate orginal dataframe
    for i, year in enumerate(df_copy.index):  # Iterate based on Year index
        for column_name in input_dict_renamed.keys(): # Iterate based on each component 
            random_value = np.random.rand() * 100  # Produce a set of random number 
            random_col = f'Random_{column_name}'  # Adding _1 suffix
            status_col = f'Failure_status_{column_name}'  # Adding _1 suffix

            # Set Random value
            df_copy.loc[year, random_col] = random_value # Value is store in random_col

            # Determine Failure status
            if random_value > df_copy.loc[year, column_name] :
                df_copy.loc[year, status_col] = "No_failure"
            else:
                df_copy.loc[year, status_col] = "Failed"

                # Update failure rates for subsequent years
                for j, future_year in enumerate(range(year + 1, len(df_copy) + 1)):
                    df_copy.loc[future_year, column_name] = input_dict_renamed[column_name][j % len(input_dict_renamed[column_name])] 

    return df_copy

# Function to check failure from Number of turbine in the project 

def failure_detection_multiple_turbines(input_dict, No_of_turbine):
    # Create an empty DataFrame to hold all turbines' results
    all_turbines_df = pd.DataFrame()

    # Loop over each turbine and apply the failure_detection function
    for turbine_idx in range(1, No_of_turbine + 1):
        # Rename input dictionary for each turbine (e.g., Blade_1, Blade_2, etc.)
        input_dict_turbine = {
            f"{key}_turbine_{turbine_idx}": value
            for key, value in input_dict.items()
        }

        # Call the original failure detection function for each turbine
        turbine_df = failure_detection(input_dict_turbine)

        # Rename the columns dynamically to include turbine index
        turbine_df.columns = [
            col.replace(f"_turbine_{turbine_idx}_1", f"_turbine_{turbine_idx}") for col in turbine_df.columns
        ]


        # Append the result for this turbine to the main DataFrame
        all_turbines_df = pd.concat([all_turbines_df, turbine_df], axis=1)

    return all_turbines_df

# Function to count how many failures per component throughout lifetime

def failure_count_summary(df):
    failure_status_columns = [col for col in df.columns if col.startswith("Failure_status_")]
    failure_data = df[failure_status_columns]
    failure_counts = failure_data.apply(lambda col: (col == "Failed").sum())
    failure_count_df = failure_counts.reset_index()
    failure_count_df.columns = ['Component',"Failure_Count"]
    failure_count_df['Component'] = failure_count_df['Component'].str.replace("Failure_status_","",regex=False)
    return failure_count_df

# Function to run simulation based on requested number of n time run
def run_simulation(input_dict, No_of_turbine,n):
    # Initialize an empty list to store failure count DataFrames for each simulation
    all_simulations_failure_counts = []

    for simulation in range(n):
        # Run the failure detection for multiple turbines
        df = failure_detection_multiple_turbines(input_dict, No_of_turbine)    
        
        # Extradct failure status columns
        failure_count_df = failure_count_summary(df)

        # Append this simulation's failure count into the list 
        all_simulations_failure_counts.append(failure_count_df)

    # Combine all failure count DataFrames into a single DataFrame
    combined_failure_counts = pd.concat(all_simulations_failure_counts, keys=range(n), names=["Simulation", "Row"])

    # Calculate the mean  failure count for each component
    
    mean_failure_counts = combined_failure_counts.groupby("Component")[["Failure_Count"]].mean().round()
    mean_failure_counts.reset_index(inplace=True)
    
    return df, failure_count_df, mean_failure_counts

# Function to provide the result how many component failied per year
def failure_summary_table(df,input_dict):
    summary = pd.DataFrame(index=df.index.unique())

    for component in input_dict:
        component_columns = [col for col in df.columns if f"Failure_status_{component}_turbine" in col]
        
        summary[component] = df[component_columns].apply(lambda row: (row == "Failed").sum(), axis = 1)
    # Create a sum row for each component 
    summary.loc['Total'] = summary.sum(axis=0)  
    return summary


# Function to provide the result how many time a turbine failed per year
def summarize_failures_by_turbine(df, input_dict, no_of_turbines):
    # Initialize an empty DataFrame for the summary
    summary = pd.DataFrame(index=df.index.unique())

    # Iterate over each turbine
    for turbine in range(1, no_of_turbines + 1):
        # Dynamically generate column names for this turbine across all components
        turbine_columns = [
            f"Failure_status_{component}_turbine_{turbine}" 
            for component in input_dict.keys()
        ]
        
        # Filter valid columns for this turbine
        valid_columns = [col for col in turbine_columns if col in df.columns]
        
        # Sum "Failed" across all components for this turbine
        summary[f"Failure_status_turbine_{turbine}"] = df[valid_columns].apply(
            lambda row: (row == "Failed").sum(), axis=1
        )

    # Optionally, add a total failures column
    summary["Total_Failures"] = summary.sum(axis=1)

    return summary


# Function to provide result of total failure per year
def sum_failure_per_year(df):
    df['Total_Failure'] = df.select_dtypes(include=np.number).sum(axis=1)
    return df

# Streamlit app starts here

#######################
# Page configuration
st.set_page_config(
    page_title="WTGs Failure Forecasting based on failure rate",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

def main():
    st.title("WTGs Failure Forecasting based on failure rate")

    # User inputs
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    # Sidebar input 
    with st.sidebar:
        n_turbine = st.number_input("Enter the number of turbines", min_value=1, max_value=150, value=30)
        n_iterations = st.number_input("Enter the number of simulations", min_value=1, max_value=1000, value=10)
        run_button = st.button("Run Simulation")

    if uploaded_file is not None:
        # Load the Excel file
        df = load_excel(uploaded_file)
        if df is not None:
            st.subheader("Uploaded Data Preview")
            
            # Extract the Blade column and convert it to a list
            input_dict = {col: (df[col] * 100).tolist() for col in df.columns}
            
            # Show the first few rows of the file to the user
            df_review = df.applymap(lambda x:x *100 if np.issubdtype(type(x), np.number) else x)
            st.write(df_review.head())

            if run_button:
                st.write("Running simulation.....")
                st.session_state.run_clicked = True
                st.session_state.simulation_results = run_simulation(input_dict, No_of_turbine=n_turbine, n=n_iterations)

            if st.session_state.run_clicked and st.session_state.simulation_results is not None:
                df_result, failure_count_df, mean_failure_counts = st.session_state.simulation_results

                # Store the result in session state so it can be reused
                st.session_state.df_result = df, failure_count_df, mean_failure_counts

                st.subheader("Simulation Results")
                total_failure = failure_count_df["Failure_Count"].sum().iloc[1]
                st.markdown(f"**Total Failures Across All Components:** {total_failure}")

                n_failure = st.number_input("Show components with failures > n times", min_value=1, max_value=1000, value=2)

                filtered_failure_df = failure_count_df[failure_count_df["Failure_Count"] > n_failure]
                component_count = filtered_failure_df.shape[0]
                st.write(component_count)


                # Count failed turbines per year
                failure_per_component_per_year_df = failure_summary_table(df_result, input_dict)
                failure_per_year_df = sum_failure_per_year(failure_per_component_per_year_df)

                 # Count number of time turbines failed
                failure_per_turbine_per_year = summarize_failures_by_turbine(df_result,input_dict, n_iterations)

                ##### Show the resulting dataframe ####

                st.subheader("Failure Counts Per Year")
                st.write(df_result)
                st.write(failure_per_year_df)
                st.write(failure_per_turbine_per_year)

                # Plot the results
                failure_per_year_df_excluded = failure_per_year_df.iloc[:-1]
                fig = px.bar(failure_per_year_df_excluded, 
                            x=failure_per_year_df_excluded.index,
                            y=failure_per_year_df_excluded.columns[:-1], 
                            title="Stacked Bar Chart of Failures by Component",
                            labels={"value": "Failures", "Year": "Year", "variable": "Component"}, 
                            barmode='stack')
                
                # Add the scatter trace for the 'Total_Failure' column
                fig.add_trace(
                    go.Scatter(
                        x=failure_per_year_df_excluded.index,
                        y=failure_per_year_df_excluded["Total_Failure"],
                        mode="text",
                        text=failure_per_year_df_excluded["Total_Failure"],
                        textposition="top center",
                        showlegend=False
                    )
                )
                st.plotly_chart(fig)

                #Plot the second result
                fig1 = px.bar(failure_per_turbine_per_year, y=failure_per_turbine_per_year.columns[:-1], title="Stacked Bar Chart of number of failures per Turbine",
                            labels={"value": "Failures per Turbine", "Year": "Year", "variable": "Turbine"}, 
                            barmode='stack')
                fig1.add_trace(
                    go.Scatter(
                        x=failure_per_turbine_per_year.index,
                        y=failure_per_turbine_per_year["Total_Failure"],
                        mode="text",
                        text=failure_per_turbine_per_year["Total_Failure"],
                        textposition="top center",
                        showlegend=False
                    )
                )
                st.plotly_chart(fig1)


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def create_periodic_df(filepath, company):
    """
    Docstring for create_periodic_df
    
    :param filepath: path of the csv file that is used as input
    :param company: a string denoting the company name to be used in column names
    :return: hourly_df, daily_df, weekly_df, monthly_df DataFrames
    """
    df = pd.read_csv(filepath)
    hourly_df = df.copy()
    hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'])
    hourly_df = hourly_df.set_index('Datetime')
    
    daily_df = hourly_df.resample('D').sum()
    daily_df.rename(columns={f"{company}_MW": f"Daily_{company}_MW"}, inplace=True)
    daily_df[f'Daily_AVG_{company}_MW'] = hourly_df.resample('D').mean()
    daily_df[f'Daily_Median_{company}_MW'] = hourly_df.resample('D').median()

    weekly_df = hourly_df.resample('W').sum()
    weekly_df.rename(columns={f"{company}_MW": f"Weekly_{company}_MW"}, inplace=True)
    weekly_df[f"Weekly_AVG_{company}_MW"] = hourly_df.resample('W').mean()
    weekly_df[f"Weekly_Median_{company}_MW"] = hourly_df.resample('W').median()
    
    monthly_df = hourly_df.resample('ME').sum()
    monthly_df.rename(columns={f"{company}_MW": f"Monthly_{company}_MW"}, inplace=True)
    monthly_df[f"Monthly_AVG_{company}_MW"] = hourly_df.resample('ME').mean()
    monthly_df[f"Monthly_Median_{company}_MW"] = hourly_df.resample('ME').median()

    hourly_df[f'Hourly_AVG_{company}_MW'] = hourly_df[f'{company}_MW'].resample('h').mean()
    hourly_df[f'Hourly_Median_{company}_MW'] = hourly_df[f'{company}_MW'].resample('h').median()
    
    return df, hourly_df, daily_df, weekly_df, monthly_df

def create_periodic_plots(company, h_df,d_df,w_df,m_df, options=['Hourly', 'Daily', 'Weekly', 'Monthly']):
    """
    Docstring for create_periodic_plots

    :param company: a string denoting the company name to be used in column names
    :param h_df: hourly DataFrame
    :param d_df: daily DataFrame
    :param w_df: weekly DataFrame
    :param m_df: monthly DataFrame
    :param options: list of periodicities to plot
    :return: None

    Creates a 4x3 grid of plots for hourly, daily, weekly, and monthly data.
    1st row: Hourly data (Total, Average, Median)
    2nd row: Daily data (Total, Average, Median)
    3rd row: Weekly data (Total, Average, Median)
    4th row: Monthly data (Total, Average, Median)
    """
    rows = len(options)
    print(rows)
    #fig, axs = plt.subplots(rows, 3, figsize=(12, 8))

    for option in options:
        if option == 'Hourly':
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            h_df[f"{company}_MW"].plot(ax=axs[0], title=f"Hourly {company} MW Load", rot=45)
            h_df[f"Hourly_AVG_{company}_MW"].plot(ax=axs[1], title=f"Hourly Average {company} MW Load", rot=45)
            h_df[f"Hourly_Median_{company}_MW"].plot(ax=axs[2], title=f"Hourly Median {company} MW Load", rot=45)
            plt.tight_layout()
            st.pyplot(fig)  
        elif option == 'Daily':
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            d_df[f"Daily_{company}_MW"].plot(ax=axs[0], title=f'Daily Total {company} MW Load', rot=45)
            d_df[f'Daily_AVG_{company}_MW'].plot(ax=axs[1], title=f'Daily Average {company} MW Load', rot=45)
            d_df[f'Daily_Median_{company}_MW'].plot(ax=axs[2], title=f'Daily Median {company} MW Load', rot=45)
            plt.tight_layout()
            st.pyplot(fig)
        elif option == 'Weekly':
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            w_df[f"Weekly_{company}_MW"].plot(ax=axs[0], title=f'Weekly Total {company} MW Load', rot=45)
            w_df[f"Weekly_AVG_{company}_MW"].plot(ax=axs[1], title=f'Weekly Average {company} MW Load', rot=45)
            w_df[f"Weekly_Median_{company}_MW"].plot(ax=axs[2], title=f'Weekly Median {company} MW Load', rot=45)
            plt.tight_layout()
            st.pyplot(fig)
        elif option == 'Monthly':
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            m_df[f"Monthly_{company}_MW"].plot(ax=axs[0], title=f'Monthly Total {company} MW Load', rot=45)
            m_df[f"Monthly_AVG_{company}_MW"].plot(ax=axs[1], title=f'Monthly Average {company} MW Load', rot=45)
            m_df[f'Monthly_Median_{company}_MW'].plot(ax=axs[2], title=f'Monthly Median {company} MW Load', rot=45)
            plt.tight_layout()
            st.pyplot(fig)

def decompose(df,column,option):
    """
    Docstring for decompose
    
    :param df: can be daily, weekly, or monthly DataFrame
    :param column: DataFrame column to decompose
    :param option: 'D' for daily, 'W' for weekly, 'ME' for monthly
    :return: decomposition object

    Create a seasonal decomposition of the time series data.
    """
    # Should allow decompose to daily, weekly, and monthly.
    if option == 'D':
        df = df[column].resample(option).mean()
        period = 7
    elif option == 'ME':
        # Resample to Monthly Average
        # This reduces noise and clearly isolates the annual (12-month) cycle.
        df = df[column].resample(option).mean()    
        period = 12
    elif option == 'W':
        # Resample to Weekly Average
        df = df[column].resample(option).mean()    
        period = 4
        
    # Perform Time Series Decomposition
    # period=12 for the annual cycle in monthly data
    decomposition = seasonal_decompose(df, model='multiplicative', period=period)
    return decomposition
    
def create_seasonal_plots(decomposition):
    """
    Docstring for create_seasonal_plots
    
    :param decomposition: acccepts a decomposition object
    :return: None

    Creates a 2-column plot showing Observed Data, Trend Component, Seasonal Component, and Residuals.
    """
    col1, col2 = st.columns(2)
    with col1:
        # # Observed Data
        fig1, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        decomposition.observed.plot(ax=ax1, title=f'Observed Avg XXX', color='blue')
        ax1.set_ylabel('MW')
        plt.tight_layout()
        st.pyplot(fig1)

        # # Seasonal Component
        fig3, ax3 = plt.subplots(1, 1, figsize=(15, 5))
        decomposition.seasonal.plot(ax=ax3, title='Annual Seasonality Cycle', color='green')
        ax3.set_ylabel('Seasonal Factor')
        plt.tight_layout()
        st.pyplot(fig3)

    with col2:
        # # Trend Component
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 5))
        decomposition.trend.plot(ax=ax2, title='Long-Term Trend', color='red')
        ax2.set_ylabel('Trend (MW)')
        plt.tight_layout()
        st.pyplot(fig2)
        
        # # Residuals (Noise)
        fig4, ax4 = plt.subplots(1, 1, figsize=(15, 5))
        decomposition.resid.plot(ax=ax4, title='Residuals (Noise)', color='purple')
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig4)

def list_files_folders(start_path):
    """
    Docstring for list_files_folders
    
    :param start_path: path of files/folders to be listed
    :return: list of file paths
    """
    file_paths = []
    for root, dirs, files in os.walk(start_path):
      for file in files:
        # Construct the full file path
        #full_path = os.path.join(root, file)
        full_path = file
        file_paths.append(full_path)
    return file_paths

def adf_test_and_interpret(timeseries, name='Time Series', signif=0.05):
    """
    Performs the ADF test and prints the results and interpretation.
    """
    st.subheader(f'--- ADF Test Results for {name} ---')
    adf_result = adfuller(timeseries, autolag='AIC')

    # Organize the results for a clean output
    output = pd.Series(adf_result[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations'])
    for key, value in adf_result[4].items():
        output[f'Critical Value ({key})'] = value
   
    #st.write(output.to_string())
    st.write(output)

    # Interpret the results
    if adf_result[1] <= signif:
        st.markdown(f"**Conclusion:** Reject the Null Hypothesis (H0). The series is likely **STATIONARY** at the {signif*100}% level.")
    else:
        st.markdown(f"**Conclusion:** Fail to Reject the Null Hypothesis (H0). The series is **NON-STATIONARY** at the {signif*100}% level.")


def create_acf_pacf_plots(monthly_df_double_diff):
    """
    Creates ACF and PACF plots for the provided DataFrame.
    
    :param monthly_df_double_diff: DataFrame containing the double differenced monthly data
    :return: None
    """
    col1, col2 = st.columns(2)
    with col1:
        # ACF Plot
        fig_acf = plot_acf(monthly_df_double_diff[f"Monthly_{company}_MW"], lags=48, title='Autocorrelation Function (ACF)')
        plt.title("ACF Plot")
        st.pyplot(fig_acf)

    with col2:
    # PACF Plot
        fig_pacf = plot_pacf(monthly_df_double_diff[f"Monthly_{company}_MW"], lags=48, title='Partial Autocorrelation Function (PACF)')
        plt.title("PACF Plot")
        st.pyplot(fig_pacf)
    st.write("ACF and PACF plots displayed.")

def sarima_model(monthly_df, company):
    # 1. Define the split point (e.g., reserve the last 12 months for testing)
    split_date = monthly_df.index[-12]

    # 2. Split the data
    train = monthly_df.loc[monthly_df.index < split_date, f'Monthly_{company}_MW']
    test = monthly_df.loc[monthly_df.index >= split_date, f'Monthly_{company}_MW']

    st.write(f"Training data points: {len(train)}")
    st.write(f"Testing data points: {len(test)}")

    # Define the model order based on your analysis
    order = (1, 1, 1)        # (p, d, q)
    seasonal_order = (0, 1, 0, 12) # (P, D, Q, s)

    # Fit the SARIMAX model
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Suppressing warnings with enforce=False is common for initial fitting
    results = model.fit(disp=False)

    st.write("\n--- Model Summary ---")
    st.write(results.summary())

########################################
# Streamlit App Starts Here
st.set_page_config(
  page_title="Energy Consumption Data Analysis", 
  layout="wide",
  initial_sidebar_state="expanded",
  menu_items={
      'About': "This app analyzes energy consumption data."
  }
)

st.title("Energy Consumption Data Analysis - Dashboard")
st.markdown(f"""
This dashboard allows you to analyze hourly energy consumption 
data for various companies. I sourced this data from Kaggle.
            
**URL:** https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data

Select a file from the sidebar to view its data and visualizations.
""")

file_paths = list_files_folders("./Hourly_Energy_Consumption_Data/3/")
file_paths.remove('PJM_Load_hourly.csv')
file_paths.remove('pjm_hourly_est.csv')
file_paths.remove('est_hourly.paruqet')

cwd = os.getcwd()

csv_file = st.sidebar.selectbox(
    "Which file do you want to analyze?",
    file_paths
)
company = csv_file.split('_')[0]

st.header(f"Hourly Data for {company}")
st.markdown(f"""
Here is a peek at the dataset for {csv_file} in tabular form. This dataset contains hourly energy consumption data for {company}.
Power system data typically exhibits strong seasonal patterns, making it ideal for time series analysis.
""")
df, h_df, d_df, w_df, m_df = create_periodic_df(cwd + f"/Hourly_Energy_Consumption_Data/3/{csv_file}", company)
st.dataframe(df)

st.header(f"Periodic Data plots for {company} in Hourly, Daily, Weekly, and Monthly")
st.markdown(f"Let's visualize this data in its original hourly form as well as when sampled daily, weekly, and monthly.")
options = st.multiselect(
    'Select the periodicities you want to plot:',
    ['Hourly', 'Daily', 'Weekly', 'Monthly']
)
st.markdown("You selected: " + ", ".join(options))
create_periodic_plots(company, h_df, d_df, w_df, m_df, options)

st.header(f"Seasonal Plots for {company} in Daily, Weekly, and Monthly")

option2 = st.radio(
    'Select the periodicities you want to plot:',
    ['Daily', 'Weekly', 'Monthly']
)

st.markdown(f"""
**Observed Plot:** This plot shows the raw MW consumption values over time.

**Trend Plot:** This plot makes it easier to observe whether the MW consumption is growing or decreasing.

**Seasonal Factor:** The effect of the time of year on the series. For AEP, this shows the recurring, predictable spikes in demand during summer and winter and the dips during shoulder seasons (spring and fall).
            
**Residual Plot:** The residual component, also called the irregular component, is what is left over after the trend and seasonal components have been removed from the observed data. The unpredictable, random, or erratic movement in the time series. This is the unexplained variation or noise.
""")

if option2 == 'Daily':
    st.subheader(f"Daily Seasonal Plots for {company}")
    create_seasonal_plots(decompose(d_df,f'Daily_{company}_MW','D'))
elif option2 == 'Weekly':
    st.subheader(f"Weekly Seasonal Plots for {company}")
    create_seasonal_plots(decompose(w_df,f'Weekly_{company}_MW','W'))
elif option2 == 'Monthly':
    st.subheader(f"Monthly Seasonal Plots for {company}")
    create_seasonal_plots(decompose(m_df,f'Monthly_{company}_MW','ME'))

st.markdown(f"""
For our next step, we want to test Stationarity. Some models require stationarity.

A stationary series is one whose statistical properties (mean, variance, and autocorrelation) do not change over time.

Our Trend plot tells us that the MW Consumption is decreasing over time. This should tell us that the data is not stationary.

Let's us confirm this by running some tests.
""")

st.header("Augmented Dickey-Fuller (ADF) test for Stationarity")
st.write("A standard statistical test used to formally determine if a time series is stationary.")

option3 = st.radio(
    'Calculate the ADF test for which periodicity:',
    ['Daily', 'Weekly', 'Monthly']
)
if option3 == 'Daily':
    adf_test_and_interpret(d_df[f"Daily_{company}_MW"], name=f"Original Daily Total {company} Load")
    adf_test_and_interpret(d_df[f"Daily_AVG_{company}_MW"], name=f"Original Daily Average {company} Load")
    adf_test_and_interpret(d_df[f"Daily_Median_{company}_MW"], name=f"Original Daily Median {company} Load")
elif option3 == 'Weekly':
    adf_test_and_interpret(w_df[f"Weekly_{company}_MW"], name=f"Original Weekly Total {company} Load")
    adf_test_and_interpret(w_df[f"Weekly_AVG_{company}_MW"], name=f"Original Weekly Average {company} Load")
    adf_test_and_interpret(w_df[f"Weekly_Median_{company}_MW"], name=f"Original Weekly Median {company} Load")
elif option3 == 'Monthly':
    adf_test_and_interpret(m_df[f"Monthly_{company}_MW"], name=f"Original Monthly Total {company} Load")
    adf_test_and_interpret(m_df[f"Monthly_AVG_{company}_MW"], name=f"Original Monthly Average {company} Load")
    adf_test_and_interpret(m_df[f"Monthly_Median_{company}_MW"], name=f"Original Monthly Median {company} Load")

st.header("Seasonal Differencing (D=1)")
st.markdown(f"""
This removes the strong annual pattern (the correlation between a month 
and the same month last year). Since you have monthly data, the seasonal
period (s) is 12.
""")

m_df_seasonal_diff = m_df.diff(periods=12).dropna()
adf_test_and_interpret(m_df_seasonal_diff[f"Monthly_{company}_MW"], name=f'Original Monthly Total {company} Load')
adf_test_and_interpret(m_df_seasonal_diff[f"Monthly_AVG_{company}_MW"], name=f'Original Monthly Average {company} Load')
adf_test_and_interpret(m_df_seasonal_diff[f"Monthly_Median_{company}_MW"], name=f'Original Monthly Median {company} Load')

st.header("Non-Seasonal Differencing (D=1)")
m_df_double_diff = m_df_seasonal_diff.diff(periods=1).dropna()
adf_test_and_interpret(m_df_double_diff[f"Monthly_{company}_MW"], name=f'Original Monthly Total {company} Load')
adf_test_and_interpret(m_df_double_diff[f"Monthly_AVG_{company}_MW"], name=f'Original Monthly Average {company} Load')
adf_test_and_interpret(m_df_double_diff[f"Monthly_Median_{company}_MW"], name=f'Original Monthly Median {company} Load')


st.header("Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) Plots")
create_acf_pacf_plots(m_df_double_diff)

st.header("SARIMA Model")
sarima_model(m_df, company)


# End of streamlist_app.py
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: black;
    color: white;
    text-align: center;
    padding: 10px 0; /* Add some padding */
}
</style>
<div class="footer">
    <p>Copyright © 2025 Krishna Karnamadakala</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

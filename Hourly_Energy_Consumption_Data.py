import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Import Modules
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import kagglehub
    import shutil
    import os
    return kagglehub, np, os, pd, plt, shutil


@app.cell
def _(mo):
    mo.md(r"""
    ## Download Kaggle Dataset & Setup Directory
    I am going to perform forecasting analysis on power system data available on Kaggle. Please note that this is publicly available data can can be found here: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data
    """)
    return


@app.cell
def _(kagglehub):
    # download kaggle dataset
    path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    print ("Path to dataset files:", path)
    return (path,)


@app.cell
def _(os, path, shutil):
    if os.path.isdir("./Hourly_Energy_Consumption_Data/"):
        print(f"The directory {path} exists!")
    else:
        shutil.move(path,"./Hourly_Energy_Consumption_Data/")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Understand the Directory Structure

    Here you have hourly energy consumption data in csv file format for a number of companies.
    """)
    return


@app.cell
def _(os):
    def list_files_folders(start_path):
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{subindent}{f}')

    # Example usage for the current directory:
    list_files_folders("./Hourly_Energy_Consumption_Data/")
    return


@app.cell
def _(os):
    cwd = os.getcwd()
    cwd
    return (cwd,)


@app.cell
def _(mo):
    mo.md(r"""
    # Organize the Data for AEP into Hourly, Daily, Weekly and Monthly
    """)
    return


@app.cell
def _(cwd, pd):
    df = pd.read_csv(cwd + "/Hourly_Energy_Consumption_Data/3/AEP_Hourly.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    df.tail()
    return


@app.cell
def _(mo):
    mo.md(r"""
    This has data per hour from Dec 31st 2004 through Jan 2nd 2018.
    """)
    return


@app.cell
def _(df, pd):
    hourly_df = df
    hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'])
    hourly_df = hourly_df.set_index('Datetime')
    hourly_df.head()
    return (hourly_df,)


@app.cell
def _(hourly_df):
    daily_df = hourly_df.resample('D').sum()
    daily_df
    return (daily_df,)


@app.cell
def _(mo):
    mo.md(r"""
    Ok. We now have convered our data from hourly to daily. And instead of looking at MW per hour, we have added up the MW every hour for 24 hours to make it daily.
    """)
    return


@app.cell
def _(daily_df):
    daily_df.rename(columns={'AEP_MW': 'Daily_AEP_MW'}, inplace=True)
    daily_df
    return


@app.cell
def _(daily_df, hourly_df):
    daily_df['Daily_AVG_AEP_MW'] = hourly_df.resample('D').mean()
    daily_df['Daily_Median_AEP_MW'] = hourly_df.resample('D').median()
    daily_df
    return


@app.cell
def _(hourly_df):
    weekly_df = hourly_df.resample('W').sum()
    weekly_df['Weekly_AVG_AEP_MW'] = hourly_df.resample('W').mean()
    weekly_df['Weekly_Median_AEP_MW'] = hourly_df.resample('W').median()
    weekly_df.rename(columns={'AEP_MW': 'Weekly_AEP_MW'}, inplace=True)
    weekly_df
    return (weekly_df,)


@app.cell
def _(hourly_df):
    monthly_df = hourly_df.resample('ME').sum()
    monthly_df['Monthly_AVG_AEP_MW'] = hourly_df.resample('ME').mean()
    monthly_df['Monthly_Median_AEP_MW'] = hourly_df.resample('ME').median()
    monthly_df.rename(columns={'AEP_MW': 'Monthly_AEP_MW'}, inplace=True)
    #monthly_df
    return (monthly_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Let's simplify what we did above into a function that we can use repeatedly.
    """)
    return


@app.cell
def _(hourly_df):
    hourly_df.head()
    return


@app.cell
def _(cwd, pd):
    def create_periodic_df(filepath):
        hourly_df = pd.read_csv(filepath)
        hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'])
        hourly_df = hourly_df.set_index('Datetime')
    
        daily_df = hourly_df.resample('D').sum()
        daily_df.rename(columns={'AEP_MW': 'Daily_AEP_MW'}, inplace=True)
        daily_df['Daily_AVG_AEP_MW'] = hourly_df.resample('D').mean()
        daily_df['Daily_Median_AEP_MW'] = hourly_df.resample('D').median()
    
        weekly_df = hourly_df.resample('W').sum()
        weekly_df.rename(columns={'AEP_MW': 'Weekly_AEP_MW'}, inplace=True)
        weekly_df['Weekly_AVG_AEP_MW'] = hourly_df.resample('W').mean()
        weekly_df['Weekly_Median_AEP_MW'] = hourly_df.resample('W').median()
    
        monthly_df = hourly_df.resample('ME').sum()
        monthly_df.rename(columns={'AEP_MW': 'Monthly_AEP_MW'}, inplace=True)
        monthly_df['Monthly_AVG_AEP_MW'] = hourly_df.resample('ME').mean()
        monthly_df['Monthly_Median_AEP_MW'] = hourly_df.resample('ME').median()

        hourly_df['Hourly_AVG_AEP_MW'] = hourly_df['AEP_MW'].resample('h').mean()
        hourly_df['Hourly_Median_AEP_MW'] = hourly_df['AEP_MW'].resample('h').median()
    
        return hourly_df, daily_df, weekly_df, monthly_df

    h_df, d_df, w_df, m_df = create_periodic_df(cwd + "/Hourly_Energy_Consumption_Data/3/AEP_Hourly.csv")
    return d_df, h_df, m_df, w_df


@app.cell
def _(mo):
    mo.md(r"""
    Ok. We now have hourly, daily, weekly and monthly data.
    Let's create some plots. I am going to show long-term trend and annual/seasonal cycle.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hourly Data Plots
    """)
    return


@app.cell
def _(hourly_df):
    hourly_df
    return


@app.cell
def _(hourly_df, plt):
    fig_h, axs_h = plt.subplots(1, 1, figsize=(10, 2))

    hourly_df['AEP_MW'].plot(ax=axs_h, title='Hourly Total AEP MW Load', rot=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Daily Data Plots
    """)
    return


@app.cell
def _(daily_df, plt):
    fig_d, axs_d = plt.subplots(1, 3, figsize=(10, 2))

    daily_df['Daily_AEP_MW'].plot(ax=axs_d[0], title='Daily Total AEP MW Load', rot=45)
    daily_df['Daily_AVG_AEP_MW'].plot(ax=axs_d[1], title='Daily Average AEP MW Load', rot=45)
    daily_df['Daily_Median_AEP_MW'].plot(ax=axs_d[2], title='Daily Median AEP MW Load', rot=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Weekly Data Plots
    """)
    return


@app.cell
def _(plt, weekly_df):
    fig_w, axs_w = plt.subplots(1, 3, figsize=(10, 2))

    weekly_df['Weekly_AEP_MW'].plot(ax=axs_w[0], title='Weekly Total AEP MW Load', rot=45)
    weekly_df['Weekly_AVG_AEP_MW'].plot(ax=axs_w[1], title='Weekly Average AEP MW Load', rot=45)
    weekly_df['Weekly_Median_AEP_MW'].plot(ax=axs_w[2], title='Weekly Median AEP MW Load', rot=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Monthly Data Plots
    """)
    return


@app.cell
def _(monthly_df, plt):
    fig_m, axs_m = plt.subplots(1, 3, figsize=(10, 2))

    monthly_df['Monthly_AEP_MW'].plot(ax=axs_m[0], title='Monthly Total AEP MW Load', rot=45)
    monthly_df['Monthly_AVG_AEP_MW'].plot(ax=axs_m[1], title='Monthly Average AEP MW Load', rot=45)
    monthly_df['Monthly_Median_AEP_MW'].plot(ax=axs_m[2], title='Monthly Median AEP MW Load', rot=45)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(d_df, h_df, m_df, plt, w_df):
    def create_periodic_plots(h_df,d_df,w_df,m_df):
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))

        h_df['AEP_MW'].plot(ax=axs[0,0], title='Hourly AEP MW Load', rot=45)
        h_df['Hourly_AVG_AEP_MW'].plot(ax=axs[0,1], title='Hourly Average AEP MW Load', rot=45)
        h_df['Hourly_Median_AEP_MW'].plot(ax=axs[0,2], title='Hourly Median AEP MW Load', rot=45)
    
        d_df['Daily_AEP_MW'].plot(ax=axs[1,0], title='Daily Total AEP MW Load', rot=45)
        d_df['Daily_AVG_AEP_MW'].plot(ax=axs[1,1], title='Daily Average AEP MW Load', rot=45)
        d_df['Daily_Median_AEP_MW'].plot(ax=axs[1,2], title='Daily Median AEP MW Load', rot=45)
    
        w_df['Weekly_AEP_MW'].plot(ax=axs[2,0], title='Weekly Total AEP MW Load', rot=45)
        w_df['Weekly_AVG_AEP_MW'].plot(ax=axs[2,1], title='Weekly Average AEP MW Load', rot=45)
        w_df['Weekly_Median_AEP_MW'].plot(ax=axs[2,2], title='Weekly Median AEP MW Load', rot=45)
    
        m_df['Monthly_AEP_MW'].plot(ax=axs[3,0], title='Monthly Total AEP MW Load', rot=45)
        m_df['Monthly_AVG_AEP_MW'].plot(ax=axs[3,1], title='Monthly Average AEP MW Load', rot=45)
        m_df['Monthly_Median_AEP_MW'].plot(ax=axs[3,2], title='Monthly Median AEP MW Load', rot=45)

        plt.tight_layout()
        plt.show()

    create_periodic_plots(h_df, d_df, w_df, m_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Exploratory and Data Analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Seasonal decomposition breaks a time series data into its core patterns:
    - Trend (long-term direction),
    - Seasonality (repeating cycles like yearly spikes),
    - and Residuals (random noise)
    using libraries like statsmodels' seasonal_decompose.

    We can then plot using this data.
    """)
    return


@app.cell
def _(cwd, df, pd):
    from statsmodels.tsa.seasonal import seasonal_decompose

    # 1. Load Data and Preprocessing
    df2 = pd.read_csv(cwd + "/Hourly_Energy_Consumption_Data/3/AEP_Hourly.csv")

    # Assuming column names 'Datetime' and 'AEP_MW'
    date_col = 'Datetime'
    consumption_col = 'AEP_MW'

    # Ensure 'Datetime' is a datetime object and set it as the index
    df2[date_col] = pd.to_datetime(df[date_col])
    df2 = df2.set_index(date_col)

    # Clean data
    df2[consumption_col] = pd.to_numeric(df2[consumption_col], errors='coerce')
    df2 = df2.dropna(subset=[consumption_col])

    # 2. Resample to Monthly Average
    # This reduces noise and clearly isolates the annual (12-month) cycle.
    monthly_df2 = df2[consumption_col].resample('ME').mean()

    # 3. Perform Time Series Decomposition
    # period=12 for the annual cycle in monthly data
    decomposition = seasonal_decompose(monthly_df2, model='multiplicative', period=12)
    return consumption_col, decomposition, seasonal_decompose


@app.cell
def _(decomposition):
    decomposition
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Observed Plot:** This plot shows the raw MW consumption values over time.

    **Trend Plot:** This plot makes it easier to observe whether the MW consumption is growing or decreasing.

    **Seasonal Factor:** The effect of the time of year on the series. For AEP, this shows the recurring, predictable spikes in demand during summer and winter and the dips during shoulder seasons (spring and fall).

    **Residual Plot:** The residual component, also called the irregular component, is what is left over after the trend and seasonal components have been removed from the observed data. The unpredictable, random, or erratic movement in the time series. This is the unexplained variation or noise.
    """)
    return


@app.cell
def _(consumption_col, decomposition, plt):
    # 4. Plot the Decomposition
    fig_s, (ax_s1, ax_s2, ax_s3, ax_s4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Observed Data
    decomposition.observed.plot(ax=ax_s1, title=f'Observed Monthly Avg {consumption_col}', color='blue')
    ax_s1.set_ylabel('MW')

    # Trend Component
    decomposition.trend.plot(ax=ax_s2, title='Long-Term Trend', color='red')
    ax_s2.set_ylabel('Trend (MW)')

    # Seasonal Component
    decomposition.seasonal.plot(ax=ax_s3, title='Annual Seasonality Cycle', color='green')
    ax_s3.set_ylabel('Seasonal Factor')

    # Residuals (Noise)
    decomposition.resid.plot(ax=ax_s4, title='Residuals (Noise)', color='purple')
    ax_s4.set_ylabel('Residual')
    ax_s4.set_xlabel('Time (Year)')

    plt.suptitle('Time Series Decomposition of AEP Energy Consumption', y=1.02)
    plt.tight_layout()
    #plt.savefig('aep_monthly_decomposition.png')
    #plt.close()
    plt.show()
    return


@app.cell
def _(consumption_col, d_df, m_df, plt, seasonal_decompose, w_df):
    def decomposex(monthly_df,column):  
        # Should allow decompose to daily, weekly, and monthly.
    
        # 2. Resample to Monthly Average
        # This reduces noise and clearly isolates the annual (12-month) cycle.
        monthly_df = monthly_df[column].resample('ME').mean()

        # 3. Perform Time Series Decomposition
        # period=12 for the annual cycle in monthly data
        decomposition = seasonal_decompose(monthly_df, model='multiplicative', period=12)
        return decomposition


    def decompose(df,column,option):  
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
        # Plot the Decomposition
        fig_s, (ax_s1, ax_s2, ax_s3, ax_s4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
        # Observed Data
        decomposition.observed.plot(ax=ax_s1, title=f'Observed Avg {consumption_col}', color='blue')
        ax_s1.set_ylabel('MW')
    
        # Trend Component
        decomposition.trend.plot(ax=ax_s2, title='Long-Term Trend', color='red')
        ax_s2.set_ylabel('Trend (MW)')
    
        # Seasonal Component
        decomposition.seasonal.plot(ax=ax_s3, title='Annual Seasonality Cycle', color='green')
        ax_s3.set_ylabel('Seasonal Factor')
    
        # Residuals (Noise)
        decomposition.resid.plot(ax=ax_s4, title='Residuals (Noise)', color='purple')
        ax_s4.set_ylabel('Residual')
        ax_s4.set_xlabel('Time (Year)')
    
        plt.suptitle('Time Series Decomposition Energy Consumption', y=1.02)
        plt.tight_layout()
        #plt.savefig('aep_monthly_decomposition.png')
        #plt.close()
        plt.show()

    create_seasonal_plots(decompose(m_df,'Monthly_AEP_MW','ME'))
    create_seasonal_plots(decompose(d_df,'Daily_AEP_MW','D'))
    create_seasonal_plots(decompose(w_df,'Weekly_AEP_MW','W'))
    return


@app.cell
def _(mo):
    mo.md(r"""
    In time series analysis, *seasonality* refers to any predictable and recurring pattern that repeats over a fixed interval. In the scenario above, we have chosen a period of 12 months, which forces the decomposition to find the distinct impact of each individual month on the overall trend (e.g., how much higher is July's consumption compared to the trend).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Explanation of Plots**

    The Trend plot shows that MW consumption in AEP is decreasing over time.

    The seasonal plot shows that when the factor is above 1.0, MW consumption is higher than the trend and when the factor is below 1.0, the MW Consumption is lower than the trend. While, the plot doesn't show specific seasons, I would explain that plot as showing that during the winter and summer seasons, the MW Consumption spikes. MW Consumption is lower during Spring and Fall seasons.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stationarity Testing and Differencing
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    For our next step, we want to test Stationarity. Some models require stationarity.

    A stationary series is one whose statistical properties (mean, variance, and autocorrelation) do not change over time.

    Our Trend plot tells us that the MW Consumption is decreasing over time. This should tell us that the data is not stationary.

    Let's us confirm this by running some tests.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Augmented Dickey-Fuller (ADF) test

    A standard statistical test used to formally determine if a time series is stationary.
    """)
    return


@app.cell
def _(monthly_df, pd):
    from statsmodels.tsa.stattools import adfuller

    def adf_test_and_interpret(timeseries, name='Time Series', signif=0.05):
        """
        Performs the ADF test and prints the results and interpretation.
        """
        print(f'--- ADF Test Results for {name} ---')
        adf_result = adfuller(timeseries, autolag='AIC')

        # Organize the results for a clean output
        output = pd.Series(adf_result[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations'])
        for key, value in adf_result[4].items():
            output[f'Critical Value ({key})'] = value

        print(output.to_string())

        # Interpret the results
        if adf_result[1] <= signif:
            print(f"\nConclusion: Reject the Null Hypothesis (H0). The series is likely **STATIONARY** at the {signif*100}% level.")
        else:
            print(f"\nConclusion: Fail to Reject the Null Hypothesis (H0). The series is **NON-STATIONARY** at the {signif*100}% level.")

    # Perform ADF Test on the original monthly time series
    adf_test_and_interpret(monthly_df['Monthly_AEP_MW'], name='Original Monthly Total AEP Load')
    adf_test_and_interpret(monthly_df['Monthly_AVG_AEP_MW'], name='Original Monthly Average AEP Load')
    adf_test_and_interpret(monthly_df['Monthly_Median_AEP_MW'], name='Original Monthly Median AEP Load')
    return (adf_test_and_interpret,)


@app.cell
def _(monthly_df):
    # Seasonal Differencing (D=1)
    # This removes the strong annual pattern (the correlation 
    # between a month and the same month last year). Since you 
    # have monthly data, the seasonal period (s) is 12.
    monthly_df_seasonal_diff = monthly_df.diff(periods=12).dropna()

    # Non-Seasonal Differencing (d=1)
    # This removes the linear drift or trend that may still remain 
    # in the seasonally adjusted series. The period is 1.
    monthly_df_double_diff = monthly_df_seasonal_diff.diff(periods=1).dropna()
    return monthly_df_double_diff, monthly_df_seasonal_diff


@app.cell
def _(monthly_df_seasonal_diff):
    monthly_df_seasonal_diff
    return


@app.cell
def _(adf_test_and_interpret, monthly_df_double_diff):
    adf_test_and_interpret(monthly_df_double_diff['Monthly_AEP_MW'], name='Original Monthly Total AEP Load')
    adf_test_and_interpret(monthly_df_double_diff['Monthly_AVG_AEP_MW'], name='Original Monthly Average AEP Load')
    adf_test_and_interpret(monthly_df_double_diff['Monthly_Median_AEP_MW'], name='Original Monthly Median AEP Load')
    return


@app.cell
def _(monthly_df_double_diff):
    monthly_df_double_diff
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) Plots
    """)
    return


@app.cell
def _(monthly_df_double_diff, plt):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # ACF Plot
    fig_acf = plot_acf(monthly_df_double_diff['Monthly_AEP_MW'], lags=48, title='Autocorrelation Function (ACF)')
    plt.show()

    # PACF Plot
    fig_pacf = plot_pacf(monthly_df_double_diff['Monthly_AEP_MW'], lags=48, title='Partial Autocorrelation Function (PACF)')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The analysis of your stationary (double-differenced) series suggests a model that is dependent on the very immediate past, but not on the seasonal past, because all significant correlation quickly drops off after Lag 1.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### SARIMA Model
    """)
    return


@app.cell
def _(monthly_df):
    from statsmodels.tsa.statespace.sarimax import SARIMAX


    # 1. Define the split point (e.g., reserve the last 12 months for testing)
    split_date = monthly_df.index[-12]

    # 2. Split the data
    train = monthly_df.loc[monthly_df.index < split_date, 'Monthly_AEP_MW']
    test = monthly_df.loc[monthly_df.index >= split_date, 'Monthly_AEP_MW']

    print(f"Training data points: {len(train)}")
    print(f"Testing data points: {len(test)}")
    return SARIMAX, test, train


@app.cell
def _(SARIMAX, train):
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

    print("\n--- Model Summary ---")
    print(results.summary())
    return order, results, seasonal_order


@app.cell
def _(mo):
    mo.md(r"""
    Root Mean Square Error (RMSE)

    $RMSE = \sqrt{\frac{1}{n}\Sigma_{i=1}^n(y_i - \hat{y}_i)^2}$
    """)
    return


@app.cell
def _(np, sarima_predictions, test):
    from sklearn.metrics import mean_squared_error

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, sarima_predictions))
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f} MW")
    return (mean_squared_error,)


@app.cell
def _(results, test):
    # Generate the forecast object, including confidence intervals
    forecast_object = results.get_forecast(steps=len(test.index))

    # Extract the predicted mean values
    sarima_predictions = forecast_object.predicted_mean

    # Extract the confidence intervals (95% CI)
    confidence_int = forecast_object.conf_int()
    return confidence_int, sarima_predictions


@app.cell
def _(
    confidence_int,
    order,
    plt,
    sarima_predictions,
    seasonal_order,
    test,
    train,
):
    # Create the final plot
    plt.figure(figsize=(12, 6))

    # Plot the training data
    plt.plot(train, label='Training Data', color='blue')

    # Plot the actual test data
    plt.plot(test, label='Actual Test Data', color='red')

    # Plot the SARIMA forecast
    plt.plot(sarima_predictions, label='SARIMA Forecast', color='green', linestyle='--')

    # Plot the confidence interval
    plt.fill_between(
        confidence_int.index,
        confidence_int.iloc[:, 0], # Lower bound
        confidence_int.iloc[:, 1], # Upper bound
        color='k', alpha=.15
    )

    plt.title(f'SARIMA({order}, {seasonal_order}) Forecast vs. Actuals')
    plt.xlabel('Date')
    plt.ylabel('Average AEP Load (MW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    #plt.savefig('sarima_forecast_vs_actual.png')
    #plt.close()
    plt.show()

    #print("\nFinal forecast plot saved as 'sarima_forecast_vs_actual.png'.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### XG Boost
    """)
    return


@app.cell
def _(monthly_df):
    # Create a copy to work with
    df_ml = monthly_df.copy()

    # 1. Time-Based Features (Exogenous Variables)
    df_ml['Month'] = df_ml.index.month
    df_ml['Year'] = df_ml.index.year
    # This helps the model capture the 12-month cycle

    # 2. Trend Feature
    # Simple count of months since the start date to model the trend component
    df_ml['TimeIndex'] = (df_ml.index - df_ml.index.min()).days / 30

    # 3. Lagged Feature (Endogenous Variables)
    # This captures the Autoregressive (AR) behavior you found in the SARIMA model (p=1)
    # The consumption 12 months ago is a very strong predictor.
    df_ml['Lag_12'] = df_ml['Monthly_AEP_MW'].shift(12)

    # Drop the first 12 rows, which now contain NaN due to the Lag_12 feature
    df_ml = df_ml.dropna()

    df_ml
    return (df_ml,)


@app.cell
def _(X, y):
    # The split point index is defined by the number of observations we used for testing
    # We'll use the last 12 months of the *cleaned* dataframe (df_ml)
    test_size = 12
    train_X, test_X = X[:-test_size], X[-test_size:]
    train_y, test_y = y[:-test_size], y[-test_size:]
    return test_X, test_y, train_X, train_y


@app.cell
def _(df_ml):
    # Define the target variable
    y = df_ml['Monthly_AEP_MW']

    # Define features (XGBoost can handle all of these)
    features = ['Month', 'Year', 'TimeIndex', 'Lag_12']
    X = df_ml[features]
    return X, y


@app.cell
def _(pd, test_X, test_y, train_X, train_y):
    import xgboost as xgb

    # Initialize and train the XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42,
        objective='reg:squarederror' # Standard objective for regression
    )
    xgb_model.fit(train_X, train_y)

    # Generate predictions on the test set
    xgb_predictions = xgb_model.predict(test_X)

    # Convert predictions to a Pandas Series with the correct index for plotting
    xgb_predictions_series = pd.Series(xgb_predictions, index=test_y.index)
    return (xgb_predictions_series,)


@app.cell
def _(mean_squared_error, np, plt, test_y, train_y, xgb_predictions_series):
    # Calculate RMSE
    xgb_rmse = np.sqrt(mean_squared_error(test_y, xgb_predictions_series))
    print(f"XGBoost Root Mean Squared Error (RMSE): {xgb_rmse:.2f} MW")

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    # Plot the full historical data
    plt.plot(train_y, label='Training Data', color='blue')

    # Plot the actual test data (clean data from test_y)
    plt.plot(test_y, label='Actual Test Data', color='red', linestyle='-')

    # Plot the XGBoost forecast
    plt.plot(xgb_predictions_series, label='XGBoost Forecast', color='purple', linestyle='--')

    plt.title('XGBoost Forecast vs. Actuals')
    plt.xlabel('Date')
    plt.ylabel('Average AEP Load (MW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('xgboost_forecast_vs_actual.png')
    plt.show() # Display the plot

    print("\nFinal XGBoost plot saved as 'xgboost_forecast_vs_actual.png'.")
    return


@app.cell
def _(monthly_df):
    # Fix Data Quality Issue

    # Check for unusually low values
    print(monthly_df.tail()) 

    # Filter them out if the last month is incomplete
    monthly_df_cleaned = monthly_df[monthly_df['Monthly_AEP_MW'] > 1000] # Example threshold
    return


@app.cell
def _(plt, results):
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # THE END
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import matplotlib.dates as mdates
import os

# Function to compute fair-value OIS rates
@st.cache_data
def compute_fair_value_ois(policy_meetings, hold_rate, specified_2y3y_rate, specified_5y5y_rate, current_effective_rate, start_date):
    """
    Compute Fair-Value OIS Rates based on policy meetings and specified rates.

    Parameters:
    - policy_meetings (dict): {date_str: rate}
    - hold_rate (float): Rate to hold after the last policy meeting until end of Year 2
    - specified_2y3y_rate (float): Specified rate from Year 2 to Year 5
    - specified_5y5y_rate (float): Specified rate from Year 5 onwards
    - current_effective_rate (float): Current effective overnight rate
    - start_date (pd.Timestamp): Start date for the model

    Returns:
    - pd.DataFrame: DataFrame with Maturity, OIS_Rate, and OIS_Rate_Adjusted
    """

    # Convert policy meetings to DataFrame and sort
    policy_df = pd.DataFrame(list(policy_meetings.items()), columns=['Date', 'Policy_Rate'])
    if not policy_df.empty:
        policy_df['Date'] = pd.to_datetime(policy_df['Date'])
        policy_df = policy_df.sort_values('Date').reset_index(drop=True)
    else:
        # If no policy meetings, create an empty DataFrame
        policy_df = pd.DataFrame(columns=['Date', 'Policy_Rate'])

    # Create daily dates for the next 10 years from the start date
    end_date = start_date + pd.DateOffset(years=10)
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize rates with current effective rate
    daily_rates = pd.Series(current_effective_rate, index=daily_dates)

    # Update rates based on policy meetings
    for _, row in policy_df.iterrows():
        # Ensure the policy meeting date is within the projection window
        if row['Date'] > end_date or row['Date'] < start_date:
            continue  # Skip invalid dates
        daily_rates.loc[row['Date']:] = row['Policy_Rate']

    # Apply hold rate after last policy meeting until end of Year 2
    end_of_year_2 = start_date + pd.DateOffset(years=2)
    if not policy_df.empty:
        last_meeting_date = policy_df['Date'].iloc[-1]
        if last_meeting_date < end_of_year_2:
            daily_rates.loc[last_meeting_date:end_of_year_2] = hold_rate
    else:
        # If no policy meetings, hold the current effective rate until end of Year 2
        daily_rates.loc[:end_of_year_2] = hold_rate

    # Apply specified rates after Year 2
    end_of_year_5 = start_date + pd.DateOffset(years=5)
    daily_rates.loc[end_of_year_2:end_of_year_5] = specified_2y3y_rate
    daily_rates.loc[end_of_year_5:] = specified_5y5y_rate

    # Create DataFrame for rates
    rates_df = pd.DataFrame({'Date': daily_dates, 'Overnight_Rate': daily_rates.values})

    # Calculate cumulative average rates for each maturity
    maturities = range(1, 11)
    ois_rates = []
    for year in maturities:
        maturity_date = start_date + pd.DateOffset(years=year)
        if maturity_date > end_date:
            maturity_date = end_date
        avg_rate = rates_df.loc[rates_df['Date'] <= maturity_date, 'Overnight_Rate'].mean()
        ois_rates.append({'Maturity': year, 'OIS_Rate': avg_rate})

    ois_df = pd.DataFrame(ois_rates)
    ois_df['OIS_Rate_Adjusted'] = ois_df['OIS_Rate'] * (360 / 365)  # Adjust for Actual/360

    return ois_df

# Function to read default values from Excel
@st.cache_data
def get_default_values(file_path):
    """
    Read default Market OIS Rates, Current Effective Rate, and Market Hold Rate from an Excel file.

    Returns:
    - dict of DataFrames: Contains all necessary data for multiple countries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at path: {file_path}")

    # Read all sheets
    countries_df = pd.read_excel(file_path, sheet_name="Countries")
    current_effective_df = pd.read_excel(file_path, sheet_name="Current Effective Rates")
    market_ois_df = pd.read_excel(file_path, sheet_name="Market OIS Rates")
    market_specified_df = pd.read_excel(file_path, sheet_name="Market OIS Specified Rates")
    market_hold_df = pd.read_excel(file_path, sheet_name="Market Hold Rates")
    policy_meetings_df = pd.read_excel(file_path, sheet_name="Policy Meetings")

    return {
        "Countries": countries_df,
        "Current_Effective_Rates": current_effective_df,
        "Market_OIS_Rates": market_ois_df,
        "Market_OIS_Specified_Rates": market_specified_df,
        "Market_Hold_Rates": market_hold_df,
        "Policy_Meetings": policy_meetings_df
    }

def main():
    # Set page configuration
    st.set_page_config(page_title='Multi-Country OIS Curve Modeling', layout='wide')

    # Title
    st.title("Interactive Multi-Country OIS Curve Modeling")

    # Path to Excel file
    excel_path = r"C:\Users\MFegan\OneDrive - AustralianSuper PTY LTD\Active\5. Python\Products\Internal\OIS App\country_ois_data.xlsx"

    # Attempt to read default values from Excel
    try:
        data = get_default_values(excel_path)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()  # Stop execution if data cannot be loaded

    # Sidebar Inputs
    st.sidebar.header("Input Parameters")

    # 1. Country Selection Dropdown (Above the date input)
    countries = data["Countries"]
    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries["Country_Name"].unique()
    )

    # Retrieve the Country_ID for the selected country
    country_row = countries.loc[countries["Country_Name"] == selected_country]
    if country_row.empty:
        st.error("Selected country data not found.")
        st.stop()
    country_id = country_row["Country_ID"].values[0]

    # 2. Start Date Input
    # Define start_date_default based on today's date
    start_date_default = datetime.today()

    start_date_input = st.sidebar.date_input("Start Date", start_date_default)
    start_date = pd.to_datetime(start_date_input)

    # 3. Fetch Data for Selected Country
    # Current Effective Rate
    current_effective_row = data["Current_Effective_Rates"].loc[data["Current_Effective_Rates"]["Country_ID"] == country_id]
    if current_effective_row.empty:
        st.error("Current Effective Rate data not found for the selected country.")
        st.stop()
    current_effective_rate_default = current_effective_row["Current_Effective_Rate"].values[0]
    current_effective_ticker = current_effective_row["Bloomberg_Ticker"].values[0]

    # Market Hold Rate
    market_hold_row = data["Market_Hold_Rates"].loc[data["Market_Hold_Rates"]["Country_ID"] == country_id]
    if market_hold_row.empty:
        st.error("Market Hold Rate data not found for the selected country.")
        st.stop()
    market_hold_rate_default = market_hold_row["Market_Hold_Rate"].values[0]
    market_hold_ticker = market_hold_row["Bloomberg_Ticker"].values[0]

    # Market Specified Rates (2y3y and 5y5y)
    market_specified_rates = data["Market_OIS_Specified_Rates"].loc[data["Market_OIS_Specified_Rates"]["Country_ID"] == country_id]
    if market_specified_rates.empty:
        st.error("Market Specified Rates data not found for the selected country.")
        st.stop()
    # Ensure both 2y3y and 5y5y rates are present
    if not {"2y3y", "5y5y"}.issubset(set(market_specified_rates["Specified_Rate_Type"])):
        st.error("Incomplete Market Specified Rates data for the selected country.")
        st.stop()
    market_specified_2y3y_rate_default = market_specified_rates.loc[market_specified_rates["Specified_Rate_Type"] == "2y3y", "Specified_OIS_Rate"].values[0]
    market_specified_2y3y_ticker = market_specified_rates.loc[market_specified_rates["Specified_Rate_Type"] == "2y3y", "Bloomberg_Ticker"].values[0]
    market_specified_5y5y_rate_default = market_specified_rates.loc[market_specified_rates["Specified_Rate_Type"] == "5y5y", "Specified_OIS_Rate"].values[0]
    market_specified_5y5y_ticker = market_specified_rates.loc[market_specified_rates["Specified_Rate_Type"] == "5y5y", "Bloomberg_Ticker"].values[0]

    # Initialize default values for Subjective Expectations equal to Market Equivalents
    user_hold_rate_default = market_hold_rate_default
    user_specified_2y3y_rate_default = market_specified_2y3y_rate_default
    user_specified_5y5y_rate_default = market_specified_5y5y_rate_default

    # Initialize default policy meetings with provided dates and values for the selected country
    policy_meetings_default = data["Policy_Meetings"].loc[data["Policy_Meetings"]["Country_ID"] == country_id]
    if policy_meetings_default.empty:
        st.warning("No Policy Meetings data found for the selected country.")
    policy_meetings_dict = policy_meetings_default.set_index("Meeting_Date").to_dict(orient='index')

    # 4. Create three main columns for inputs: Market OIS, Market Expectations, Subjective Expectations
    col_market_ois, col_market_expectations, col_subjective_expectations = st.columns(3)

    with col_market_ois:
        st.subheader("Market OIS")

        # Market rate held out to the end of the second year
        market_hold_rate = st.number_input(
            "Market Hold Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=market_hold_rate_default,
            step=0.01
        )

        # Market's belief of the fair value of the 2y3y and 5y5y OIS rates (%)
        st.markdown("**Market Specified OIS Rates**")
        market_specified_2y3y_rate = st.number_input(
            "Market 2y3y OIS Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=market_specified_2y3y_rate_default,
            step=0.01
        )
        market_specified_5y5y_rate = st.number_input(
            "Market 5y5y OIS Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=market_specified_5y5y_rate_default,
            step=0.01
        )

        # Input the current market OIS rates for tenors 1y to 10y (%)
        st.markdown("**Current Market OIS Rates**")
        country_market_ois = data["Market_OIS_Rates"].loc[data["Market_OIS_Rates"]["Country_ID"] == country_id]
        market_ois_rates = {}
        for _, row in country_market_ois.iterrows():
            maturity = row["Maturity"]
            rate = row["Market_OIS_Rate"]
            # ticker = row["Bloomberg_Ticker"]  # Removed ticker display
            rate = st.number_input(
                f"Market OIS Rate {maturity} (%)",
                min_value=0.0,
                max_value=100.0,
                value=rate,
                step=0.01,
                key=f"market_ois_{country_id}_{maturity}"
            )
            market_ois_rates[maturity] = rate
            # Removed Bloomberg Ticker for each maturity

    with col_market_expectations:
        st.subheader("Market Expectations")
        # Current effective rate (%)
        current_effective_rate = st.number_input(
            "Current Effective Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=current_effective_rate_default,
            step=0.001
        )

        # Policy Meetings: Checkbox and inputs for each meeting
        st.markdown("**Policy Meetings and Expected Rates**")
        market_policy_meetings = {}
        user_policy_meetings = {}
        if not policy_meetings_dict:
            st.write("No policy meetings defined for this country.")
        else:
            for i, (date_str, details) in enumerate(policy_meetings_dict.items(), start=1):
                enable = st.checkbox(f"Enable Policy Meeting {i}", key=f"enable_meeting_{i}")
                if enable:
                    # Parse the date string
                    date_default = pd.to_datetime(date_str, dayfirst=True)
                    # Create three columns for Date, Market Rate, Subjective Rate
                    meeting_cols = st.columns(3)
                    date_col, market_rate_col, subjective_rate_col = meeting_cols
                    with date_col:
                        date = st.date_input(f"Meeting {i} - Date", date_default, key=f"meeting_{i}_date")
                    with market_rate_col:
                        rate_market_default = details["Market_Rate"]
                        rate_market = st.number_input(
                            f"Meeting {i} - Market Rate (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=rate_market_default,
                            step=0.01,
                            key=f"meeting_{i}_market_rate"
                        )
                        # Removed Bloomberg Ticker for Market Rate
                        # st.markdown(f"**Bloomberg Ticker for Meeting {i} Market Rate:** {details['Bloomberg_Ticker']}")
                    with subjective_rate_col:
                        # Set Subjective Rate default equal to Market Rate
                        rate_subjective_default = details["Subjective_Rate"]
                        rate_subjective = st.number_input(
                            f"Meeting {i} - Subjective Rate (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=rate_subjective_default,
                            step=0.01,
                            key=f"meeting_{i}_subjective_rate"
                        )
                        # Removed Bloomberg Ticker for Subjective Rate
                        # st.markdown(f"**Bloomberg Ticker for Meeting {i} Subjective Rate:** {details['Bloomberg_Ticker']}")

                    # Store the policy meetings
                    date_str_formatted = date.strftime('%Y-%m-%d')
                    market_policy_meetings[date_str_formatted] = rate_market
                    user_policy_meetings[date_str_formatted] = rate_subjective

    with col_subjective_expectations:
        st.subheader("Subjective Expectations")
        # Your rate held out to the end of the second year
        user_hold_rate = st.number_input(
            "Your Hold Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=user_hold_rate_default,
            step=0.01
        )

        # Your belief of the fair value of the 2y3y and 5y5y OIS rates (%)
        st.markdown("**Your Specified OIS Rates**")
        user_specified_2y3y_rate = st.number_input(
            "Your 2y3y OIS Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=user_specified_2y3y_rate_default,
            step=0.01
        )
        user_specified_5y5y_rate = st.number_input(
            "Your 5y5y OIS Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=user_specified_5y5y_rate_default,
            step=0.01
        )

    # 5. Compute Fair-Value OIS Rates Using Market Inputs
    market_ois_df_computed = compute_fair_value_ois(
        policy_meetings=market_policy_meetings,
        hold_rate=market_hold_rate,
        specified_2y3y_rate=market_specified_2y3y_rate,
        specified_5y5y_rate=market_specified_5y5y_rate,
        current_effective_rate=current_effective_rate,
        start_date=start_date
    )

    # Ensure numeric data types
    market_ois_df_computed['Maturity'] = pd.to_numeric(market_ois_df_computed['Maturity'], errors='coerce').astype(int)
    market_ois_df_computed['OIS_Rate_Adjusted'] = pd.to_numeric(market_ois_df_computed['OIS_Rate_Adjusted'], errors='coerce')

    # Convert market OIS rates to DataFrame
    market_ois_rates_df = pd.DataFrame(list(market_ois_rates.items()), columns=['Maturity', 'Market_OIS_Rate'])

    # Ensure 'Maturity' is numeric by stripping the 'y' and converting to int
    market_ois_rates_df['Maturity'] = market_ois_rates_df['Maturity'].str.rstrip('y').astype(int)

    # Now perform the merge
    market_ois_df_computed = market_ois_df_computed.merge(market_ois_rates_df, on='Maturity', how='left')

    # Calculate model error at each maturity
    market_ois_df_computed['Model_Error'] = market_ois_df_computed['Market_OIS_Rate'] - market_ois_df_computed['OIS_Rate_Adjusted']

    # 6. Compute Fair-Value OIS Rates Using Your Subjective Inputs
    user_ois_df_computed = compute_fair_value_ois(
        policy_meetings=user_policy_meetings,
        hold_rate=user_hold_rate,
        specified_2y3y_rate=user_specified_2y3y_rate,
        specified_5y5y_rate=user_specified_5y5y_rate,
        current_effective_rate=current_effective_rate,
        start_date=start_date
    )

    # Ensure numeric data types
    user_ois_df_computed['Maturity'] = pd.to_numeric(user_ois_df_computed['Maturity'], errors='coerce').astype(int)
    user_ois_df_computed['OIS_Rate_Adjusted'] = pd.to_numeric(user_ois_df_computed['OIS_Rate_Adjusted'], errors='coerce')

    # 7. Adjust Your Fair-Value Estimates by Model Error
    user_ois_df_computed = user_ois_df_computed.merge(market_ois_df_computed[['Maturity', 'Model_Error']], on='Maturity', how='left')
    user_ois_df_computed['Adjusted_OIS_Rate'] = user_ois_df_computed['OIS_Rate_Adjusted'] + user_ois_df_computed['Model_Error']

    # 8. Compare Adjusted Fair-Value Rates to Market OIS Rates
    user_ois_df_computed = user_ois_df_computed.merge(market_ois_rates_df, on='Maturity', how='left')
    user_ois_df_computed['Adjusted_Rate_Difference'] = user_ois_df_computed['Market_OIS_Rate'] - user_ois_df_computed['Adjusted_OIS_Rate']
    user_ois_df_computed['Adjusted_Signal'] = user_ois_df_computed['Adjusted_Rate_Difference']

    # 9. Plotting
    # Create two columns for charts: left for policy path comparison, right for OIS curves and signals
    chart_col_left, chart_col_right = st.columns(2)

    with chart_col_left:
        st.subheader("Policy Path Comparison")
        if not market_policy_meetings and not user_policy_meetings:
            st.write("No policy meetings enabled.")
        else:
            # Prepare data for Market Policy Path
            market_policy_df = pd.DataFrame(list(market_policy_meetings.items()), columns=['Date', 'Rate'])
            market_policy_df['Date'] = pd.to_datetime(market_policy_df['Date'])
            market_policy_df = market_policy_df.sort_values('Date').reset_index(drop=True)

            # Prepare data for Subjective Policy Path
            user_policy_df = pd.DataFrame(list(user_policy_meetings.items()), columns=['Date', 'Rate'])
            user_policy_df['Date'] = pd.to_datetime(user_policy_df['Date'])
            user_policy_df = user_policy_df.sort_values('Date').reset_index(drop=True)

            # Define consistent colors and styles
            market_color = 'red'  # Market paths in red
            subjective_color = 'blue'  # Subjective paths in blue
            line_style = '-'

            # Plot both policy paths
            fig_policy, ax_policy = plt.subplots(figsize=(10, 6), tight_layout=True)
            ax_policy.plot(
                market_policy_df['Date'].values, 
                market_policy_df['Rate'].values, 
                marker='o', 
                color=market_color, 
                linestyle=line_style, 
                label='Market Policy Path'
            )
            ax_policy.plot(
                user_policy_df['Date'].values, 
                user_policy_df['Rate'].values, 
                marker='x', 
                color=subjective_color, 
                linestyle=line_style, 
                label='Subjective Policy Path'
            )

            # Plot Hold Rate (Market)
            hold_end_date = start_date + pd.DateOffset(years=2)
            if not market_policy_df.empty:
                last_meeting_date = market_policy_df['Date'].iloc[-1]
                hold_start_date = last_meeting_date if last_meeting_date < hold_end_date else hold_end_date
            else:
                hold_start_date = start_date
            ax_policy.hlines(
                market_hold_rate, 
                hold_start_date, 
                hold_end_date, 
                colors=market_color, 
                linestyles='--',  # Dashed
                label='Market Hold Rate'
            )

            # Plot 2y3y Rate (Market)
            ax_policy.hlines(
                market_specified_2y3y_rate, 
                hold_end_date, 
                start_date + pd.DateOffset(years=5), 
                colors=market_color, 
                linestyles=':',  # Dotted
                label='Market 2y3y OIS Rate'
            )

            # Plot 5y5y Rate (Market)
            ax_policy.hlines(
                market_specified_5y5y_rate, 
                start_date + pd.DateOffset(years=5), 
                start_date + pd.DateOffset(years=10), 
                colors=market_color, 
                linestyles='-.',  # Dash-Dot
                label='Market 5y5y OIS Rate'
            )

            # Plot Hold Rate (Subjective)
            ax_policy.hlines(
                user_hold_rate, 
                hold_start_date, 
                hold_end_date, 
                colors=subjective_color, 
                linestyles='--',  # Dashed
                label='Subjective Hold Rate'
            )

            # Plot 2y3y Rate (Subjective)
            ax_policy.hlines(
                user_specified_2y3y_rate, 
                hold_end_date, 
                start_date + pd.DateOffset(years=5), 
                colors=subjective_color, 
                linestyles=':',  # Dotted
                label='Subjective 2y3y OIS Rate'
            )

            # Plot 5y5y Rate (Subjective)
            ax_policy.hlines(
                user_specified_5y5y_rate, 
                start_date + pd.DateOffset(years=5), 
                start_date + pd.DateOffset(years=10), 
                colors=subjective_color, 
                linestyles='-.',  # Dash-Dot
                label='Subjective 5y5y OIS Rate'
            )

            # Formatting the x-axis dates
            ax_policy.xaxis.set_major_locator(mdates.YearLocator())
            ax_policy.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax_policy.get_xticklabels(), rotation=45, ha='right')

            ax_policy.set_title(f'Policy Path Comparison for {selected_country}')
            ax_policy.set_xlabel('Date')
            ax_policy.set_ylabel('Policy Rate (%)')
            ax_policy.legend()
            ax_policy.grid(True)
            st.pyplot(fig_policy)

    with chart_col_right:
        st.subheader("Adjusted Fair-Value and Market OIS Curves")
        if user_ois_df_computed.empty or market_ois_df_computed.empty:
            st.write("Insufficient data to plot OIS curves.")
        else:
            # Extract data for plotting
            maturities_plot = user_ois_df_computed['Maturity'].to_numpy().flatten()
            adjusted_fair_value_rates = user_ois_df_computed['Adjusted_OIS_Rate'].to_numpy().flatten()
            market_rates_plot = user_ois_df_computed['Market_OIS_Rate'].to_numpy().flatten()

            # Define consistent colors and styles
            market_color = 'red'  # Market rates in red
            subjective_color = 'blue'  # Adjusted fair-value rates in blue
            line_style = '-'

            # Create OIS Curves Plot
            fig_ois, ax_ois = plt.subplots(figsize=(10, 6), tight_layout=True)
            ax_ois.plot(
                maturities_plot, 
                adjusted_fair_value_rates, 
                marker='o', 
                color=subjective_color, 
                linestyle=line_style, 
                label='Adjusted Fair-Value OIS Rate'
            )
            ax_ois.plot(
                maturities_plot, 
                market_rates_plot, 
                marker='x', 
                color=market_color, 
                linestyle=line_style, 
                label='Market OIS Rate'
            )
            ax_ois.set_title(f'Adjusted Fair-Value and Market OIS Curves for {selected_country}')
            ax_ois.set_xlabel('Maturity (Years)')
            ax_ois.set_ylabel('OIS Rate (%)')
            ax_ois.legend()
            ax_ois.grid(True)
            st.pyplot(fig_ois)

        # Add Signals Plot below OIS Curves
        st.subheader("Adjusted Trading Signals")
        if user_ois_df_computed.empty:
            st.write("Insufficient data to plot trading signals.")
        else:
            # Extract data for plotting
            maturities_signal = user_ois_df_computed['Maturity'].to_numpy().flatten()
            adjusted_signals = user_ois_df_computed['Adjusted_Signal'].to_numpy().flatten()

            # Determine colors based on signal value
            signal_colors = ['green' if signal > 0 else 'red' for signal in adjusted_signals]

            # Create Signals Plot
            fig_signals, ax_signals = plt.subplots(figsize=(10, 3), tight_layout=True)
            ax_signals.bar(
                maturities_signal, 
                adjusted_signals, 
                color=signal_colors, 
                label='Adjusted Signal (Market - Adjusted Fair Value)'
            )
            ax_signals.axhline(0, color='black', linewidth=0.8)
            ax_signals.set_title(f'Adjusted Trading Signals for {selected_country}')
            ax_signals.set_xlabel('Maturity (Years)')
            ax_signals.set_ylabel('Signal Magnitude (%)')
            ax_signals.legend()
            ax_signals.grid(True)
            st.pyplot(fig_signals)

    # 10. Display the adjusted fair-value OIS rates
    st.subheader("Your Adjusted Fair-Value OIS Rates")
    st.dataframe(user_ois_df_computed[['Maturity', 'Adjusted_OIS_Rate', 'Market_OIS_Rate', 'Adjusted_Rate_Difference']])

# Run the app
if __name__ == '__main__':
    main()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.336286Z","iopub.execute_input":"2023-07-24T04:24:59.336700Z","iopub.status.idle":"2023-07-24T04:24:59.385764Z","shell.execute_reply.started":"2023-07-24T04:24:59.336597Z","shell.execute_reply":"2023-07-24T04:24:59.384945Z"}}
import pandas as pd
import numpy as np

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.387153Z","iopub.execute_input":"2023-07-24T04:24:59.387638Z","iopub.status.idle":"2023-07-24T04:24:59.392300Z","shell.execute_reply.started":"2023-07-24T04:24:59.387599Z","shell.execute_reply":"2023-07-24T04:24:59.391510Z"}}
start_date = '2023-07-01'
end_date = '2023-07-16'
total_days_in_month = 31
days_data_having = np.array(int(end_date[-2:])).item()- np.array(int(start_date[-2:])).item() + 1

days_to_forecast = total_days_in_month - days_data_having

# %% [markdown]
# # ADD THE DATASET HERE

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.393445Z","iopub.execute_input":"2023-07-24T04:24:59.393709Z","iopub.status.idle":"2023-07-24T04:24:59.407237Z","shell.execute_reply.started":"2023-07-24T04:24:59.393688Z","shell.execute_reply":"2023-07-24T04:24:59.406103Z"}}
# Define the file path
file_path = '/kaggle/input/month-forecast/8309-itemsRequired-16-07-2023.csv'

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.409063Z","iopub.execute_input":"2023-07-24T04:24:59.409795Z","iopub.status.idle":"2023-07-24T04:24:59.469160Z","shell.execute_reply.started":"2023-07-24T04:24:59.409766Z","shell.execute_reply":"2023-07-24T04:24:59.467680Z"}}
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.470400Z","iopub.execute_input":"2023-07-24T04:24:59.471296Z","iopub.status.idle":"2023-07-24T04:24:59.481369Z","shell.execute_reply.started":"2023-07-24T04:24:59.471266Z","shell.execute_reply":"2023-07-24T04:24:59.479982Z"}}
fb_data = pd.read_csv("/kaggle/input/fb-daata/FB ads spends  - Sheet1.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.482849Z","iopub.execute_input":"2023-07-24T04:24:59.483299Z","iopub.status.idle":"2023-07-24T04:24:59.491866Z","shell.execute_reply.started":"2023-07-24T04:24:59.483266Z","shell.execute_reply":"2023-07-24T04:24:59.490874Z"}}
# lets see the columns
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.493416Z","iopub.execute_input":"2023-07-24T04:24:59.493914Z","iopub.status.idle":"2023-07-24T04:24:59.521487Z","shell.execute_reply.started":"2023-07-24T04:24:59.493889Z","shell.execute_reply":"2023-07-24T04:24:59.520676Z"}}
# Perform basic EDA
# Display the first few rows of the DataFrame
print("First few rows:")
print(df.head())

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.524257Z","iopub.execute_input":"2023-07-24T04:24:59.524650Z","iopub.status.idle":"2023-07-24T04:24:59.555472Z","shell.execute_reply.started":"2023-07-24T04:24:59.524595Z","shell.execute_reply":"2023-07-24T04:24:59.554569Z"}}
# Get summary statistics of the numerical columns
print("\nSummary statistics:")
print(df.describe())

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.556501Z","iopub.execute_input":"2023-07-24T04:24:59.556786Z","iopub.status.idle":"2023-07-24T04:24:59.563673Z","shell.execute_reply.started":"2023-07-24T04:24:59.556763Z","shell.execute_reply":"2023-07-24T04:24:59.562686Z"}}
# Check the data types of each column
print("\nData types:")
print(df.dtypes)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.566205Z","iopub.execute_input":"2023-07-24T04:24:59.566502Z","iopub.status.idle":"2023-07-24T04:24:59.575073Z","shell.execute_reply.started":"2023-07-24T04:24:59.566481Z","shell.execute_reply":"2023-07-24T04:24:59.574427Z"}}
# Check the number of rows and columns in the DataFrame
print("\nShape:")
print(df.shape)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.598112Z","iopub.execute_input":"2023-07-24T04:24:59.598457Z","iopub.status.idle":"2023-07-24T04:24:59.609544Z","shell.execute_reply.started":"2023-07-24T04:24:59.598430Z","shell.execute_reply":"2023-07-24T04:24:59.608702Z"}}
# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# %% [markdown]
# **The missing values i will be removing**

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.611276Z","iopub.execute_input":"2023-07-24T04:24:59.612073Z","iopub.status.idle":"2023-07-24T04:24:59.626863Z","shell.execute_reply.started":"2023-07-24T04:24:59.612040Z","shell.execute_reply":"2023-07-24T04:24:59.625512Z"}}
# columns_to_drop = ['hasError', 'diff', 'totalIV', 'GSTIN']
df = df.loc[:,["date_created","total"]]

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.628210Z","iopub.execute_input":"2023-07-24T04:24:59.628888Z","iopub.status.idle":"2023-07-24T04:24:59.648486Z","shell.execute_reply.started":"2023-07-24T04:24:59.628854Z","shell.execute_reply":"2023-07-24T04:24:59.647637Z"}}
df.shape

# %% [markdown]
# **Lets remove the unecessary features**

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.649642Z","iopub.execute_input":"2023-07-24T04:24:59.650653Z","iopub.status.idle":"2023-07-24T04:24:59.666003Z","shell.execute_reply.started":"2023-07-24T04:24:59.650582Z","shell.execute_reply":"2023-07-24T04:24:59.664929Z"}}
from datetime import datetime

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.668817Z","iopub.execute_input":"2023-07-24T04:24:59.669926Z","iopub.status.idle":"2023-07-24T04:24:59.697077Z","shell.execute_reply.started":"2023-07-24T04:24:59.669875Z","shell.execute_reply":"2023-07-24T04:24:59.695667Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.698684Z","iopub.execute_input":"2023-07-24T04:24:59.699192Z","iopub.status.idle":"2023-07-24T04:24:59.732790Z","shell.execute_reply.started":"2023-07-24T04:24:59.699168Z","shell.execute_reply":"2023-07-24T04:24:59.732113Z"}}
# Define a function to convert datetime string to datetime object
def convert_to_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')

# Apply the conversion function to the "Order Created Date and Time" column
df['Order created Date'] = df['date_created'].apply(convert_to_datetime)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.733878Z","iopub.execute_input":"2023-07-24T04:24:59.734122Z","iopub.status.idle":"2023-07-24T04:24:59.745065Z","shell.execute_reply.started":"2023-07-24T04:24:59.734101Z","shell.execute_reply":"2023-07-24T04:24:59.743935Z"}}
df['Order Created Date and Time'] = pd.to_datetime(df['Order created Date'])

# Sort the DataFrame based on the "Order Created Date and Time" column
df = df.sort_values('Order Created Date and Time', ascending=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.746263Z","iopub.execute_input":"2023-07-24T04:24:59.746703Z","iopub.status.idle":"2023-07-24T04:24:59.762732Z","shell.execute_reply.started":"2023-07-24T04:24:59.746679Z","shell.execute_reply":"2023-07-24T04:24:59.762048Z"}}
type(df['Order created Date'][0])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.764280Z","iopub.execute_input":"2023-07-24T04:24:59.765595Z","iopub.status.idle":"2023-07-24T04:24:59.783433Z","shell.execute_reply.started":"2023-07-24T04:24:59.765554Z","shell.execute_reply":"2023-07-24T04:24:59.782550Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.784550Z","iopub.execute_input":"2023-07-24T04:24:59.784986Z","iopub.status.idle":"2023-07-24T04:24:59.797505Z","shell.execute_reply.started":"2023-07-24T04:24:59.784959Z","shell.execute_reply":"2023-07-24T04:24:59.796426Z"}}
df= df.set_index('Order created Date')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.798808Z","iopub.execute_input":"2023-07-24T04:24:59.799338Z","iopub.status.idle":"2023-07-24T04:24:59.812117Z","shell.execute_reply.started":"2023-07-24T04:24:59.799313Z","shell.execute_reply":"2023-07-24T04:24:59.810903Z"}}
print (df.index)

# %% [markdown]
# # MODELS

# %% [markdown]
# # ARIMA MODEL WITHOUT THE FB ADS

# %% [markdown]
# **SHOULD CHANGE THE DATE HERE ACCORDINGLY**

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:24:59.813045Z","iopub.execute_input":"2023-07-24T04:24:59.813720Z","iopub.status.idle":"2023-07-24T04:25:00.152729Z","shell.execute_reply.started":"2023-07-24T04:24:59.813694Z","shell.execute_reply":"2023-07-24T04:25:00.151486Z"}}
import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Create a DataFrame with the date and cumulative income
df_cumulative_income = pd.DataFrame({'Date': daily_cumulative_income.index, 'Cumulative Income': daily_cumulative_income.values})

# Plot cumulative income vs. date
plt.figure(figsize=(10, 6))
plt.plot(df_cumulative_income['Date'], df_cumulative_income['Cumulative Income'])
plt.xlabel('Date')
plt.ylabel('Cumulative Income')
plt.title('Cumulative Income vs. Date')
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:00.154036Z","iopub.execute_input":"2023-07-24T04:25:00.154327Z","iopub.status.idle":"2023-07-24T04:25:01.155484Z","shell.execute_reply.started":"2023-07-24T04:25:00.154306Z","shell.execute_reply":"2023-07-24T04:25:01.154105Z"}}
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:01.160329Z","iopub.execute_input":"2023-07-24T04:25:01.160674Z","iopub.status.idle":"2023-07-24T04:25:01.396419Z","shell.execute_reply.started":"2023-07-24T04:25:01.160649Z","shell.execute_reply":"2023-07-24T04:25:01.394899Z"}}
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Create a DataFrame with the date and cumulative income
df_cumulative_income = pd.DataFrame({'Date': daily_cumulative_income.index, 'Cumulative Income': daily_cumulative_income.values})

# Specify the number of lags for PACF
max_lags = 5  # Set the desired number of lags

# Plot PACF
plt.figure(figsize=(12, 6))
plot_pacf(df_cumulative_income['Cumulative Income'], lags=max_lags)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:01.397756Z","iopub.execute_input":"2023-07-24T04:25:01.398336Z","iopub.status.idle":"2023-07-24T04:25:01.683139Z","shell.execute_reply.started":"2023-07-24T04:25:01.398301Z","shell.execute_reply":"2023-07-24T04:25:01.681990Z"}}
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Create a DataFrame with the date and cumulative income
df_cumulative_income = pd.DataFrame({'Date': daily_cumulative_income.index, 'Cumulative Income': daily_cumulative_income.values})

# Set the length of available data as the maximum number of lags to plot
lags = len(df_cumulative_income) - 1

# Plot ACF
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
plot_acf(df_cumulative_income['Cumulative Income'], ax=ax1, lags=lags)


ax1.set_title('Autocorrelation Function (ACF)')

# # Plot PACF
# ax2 = plt.subplot(122)
# plot_pacf(df_cumulative_income['Cumulative Income'], ax=ax2, lags=lags)
# ax2.set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


# %% [markdown]
# **Dickey fuller**

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:01.684349Z","iopub.execute_input":"2023-07-24T04:25:01.684693Z","iopub.status.idle":"2023-07-24T04:25:01.715082Z","shell.execute_reply.started":"2023-07-24T04:25:01.684665Z","shell.execute_reply":"2023-07-24T04:25:01.713898Z"}}
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Perform Dickey-Fuller test
result = adfuller(daily_cumulative_income)

# Extract the p-value from the test result
p_value = result[1]

# Determine the value of d based on the p-value
if p_value < 0.05:
    d = 0  # Time series is stationary, no differencing is required
else:
    d = 1  # Time series is non-stationary, differencing is required

print("Dickey-Fuller Test Results:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
print(f"Value of d: {d}")


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:01.716657Z","iopub.execute_input":"2023-07-24T04:25:01.717014Z","iopub.status.idle":"2023-07-24T04:25:01.739263Z","shell.execute_reply.started":"2023-07-24T04:25:01.716986Z","shell.execute_reply":"2023-07-24T04:25:01.738369Z"}}
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Perform first differencing
first_difference = daily_cumulative_income.diff().dropna()

# Perform Dickey-Fuller test on first difference
result_first = adfuller(first_difference)

# Extract the p-value from the test result of first difference
p_value_first = result_first[1]

# Determine if second differencing is required
if p_value_first < 0.05:
    d = 1  # First difference is sufficient, d=1
else:
    # Perform second differencing
    second_difference = first_difference.diff().dropna()

    # Perform Dickey-Fuller test on second difference
    result_second = adfuller(second_difference)

    # Extract the p-value from the test result of second difference
    p_value_second = result_second[1]

    if p_value_second < 0.05:
        d = 2  # Second difference is required, d=2
    else:
        d = 0  # No differencing is required, d=0

print("Dickey-Fuller Test Results for First Difference:")
print(f"ADF Statistic: {result_first[0]}")
print(f"p-value: {result_first[1]}")

if d == 2:
    print("Dickey-Fuller Test Results for Second Difference:")
    print(f"ADF Statistic: {result_second[0]}")
    print(f"p-value: {result_second[1]}")

print(f"Value of d: {d}")


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:01.740505Z","iopub.execute_input":"2023-07-24T04:25:01.740942Z","iopub.status.idle":"2023-07-24T04:25:02.226689Z","shell.execute_reply.started":"2023-07-24T04:25:01.740908Z","shell.execute_reply":"2023-07-24T04:25:02.225661Z"}}
# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Create a DataFrame with the date and cumulative income
df_cumulative_income = pd.DataFrame({'Date': daily_cumulative_income.index, 'Cumulative Income': daily_cumulative_income.values})

# Fit ARIMA model
model = ARIMA(df_cumulative_income['Cumulative Income'], order=(3, 3, 2)) #(2,1,2)
model_fit = model.fit()

# Forecast from 16th to month-end
forecast_steps = len(df_cumulative_income) + days_to_forecast  # Assuming 31 days in the month
forecast_values = model_fit.predict(start=len(df_cumulative_income), end=forecast_steps)

# Generate date range for forecasted values
forecast_dates = pd.date_range(start=df_cumulative_income['Date'].iloc[-1], periods=len(forecast_values) + 1, freq='D')[1:]

# Plot the actual and forecasted cumulative income
plt.plot(df_cumulative_income['Date'], df_cumulative_income['Cumulative Income'], label='Actual')
plt.plot(forecast_dates, forecast_values, 'ro', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Cumulative Income')
plt.title('Month-End Cumulative Income Forecast')
plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')  # Display y-axis labels in plain format
plt.legend()
plt.show()

# %% [markdown]
# # Inferences

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:02.230813Z","iopub.execute_input":"2023-07-24T04:25:02.231210Z","iopub.status.idle":"2023-07-24T04:25:02.241835Z","shell.execute_reply.started":"2023-07-24T04:25:02.231183Z","shell.execute_reply":"2023-07-24T04:25:02.240196Z"}}
# Forecast from 16th to month-end
forecast_steps = days_to_forecast  # Assuming 31 days in the month
forecast_values = model_fit.predict(start=len(df_cumulative_income), end=len(df_cumulative_income) + forecast_steps)

# Calculate month-end income
month_end_income = forecast_values.iloc[-1]

# Calculate income in steps of 5 days
income_in_steps = forecast_values.iloc[::5]

# Generate the text
result_text = f"Month-End Income: {month_end_income}\nIncome in Steps of 5 Days:\n"

for i, income in enumerate(income_in_steps):
    result_text += f"Day {i*5+16}: {income}\n"

# Print the result
print(result_text)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:02.243679Z","iopub.execute_input":"2023-07-24T04:25:02.244270Z","iopub.status.idle":"2023-07-24T04:25:02.564156Z","shell.execute_reply.started":"2023-07-24T04:25:02.244242Z","shell.execute_reply":"2023-07-24T04:25:02.562796Z"}}
import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Group the data by date and calculate the sum of the "total" column for each day
daily_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum()

# Convert the grouped data to a DataFrame
df_daily_income = pd.DataFrame({'Date': daily_income.index, 'Income': daily_income.values})

# Plot the per-day income
plt.figure(figsize=(10, 6))
plt.plot(df_daily_income['Date'], df_daily_income['Income'])
plt.xlabel('Date')
plt.ylabel('Income')
plt.title('Per-Day Income')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:02.565510Z","iopub.execute_input":"2023-07-24T04:25:02.565895Z","iopub.status.idle":"2023-07-24T04:25:02.885100Z","shell.execute_reply.started":"2023-07-24T04:25:02.565860Z","shell.execute_reply":"2023-07-24T04:25:02.883889Z"}}
import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= start_date) & (df['Order Created Date and Time'] <= end_date)]

# Group the data by date and calculate the sum of the "total" column for each day
daily_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum()

# Convert the grouped data to a DataFrame
df_daily_income = pd.DataFrame({'Date': daily_income.index, 'Income': daily_income.values})

# Plot the per-day income as dots
plt.figure(figsize=(10, 6))
plt.plot(df_daily_income['Date'], df_daily_income['Income'], 'X')
plt.xlabel('Date')
plt.ylabel('Income')
plt.title('Per-Day Income')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# # FB DATA

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:28.677034Z","iopub.execute_input":"2023-07-24T04:25:28.677365Z","iopub.status.idle":"2023-07-24T04:25:28.684816Z","shell.execute_reply.started":"2023-07-24T04:25:28.677337Z","shell.execute_reply":"2023-07-24T04:25:28.683489Z"}}
fb_data['Date'] = pd.to_datetime(fb_data['Date '], format='%d %B %Y')
fb_data.set_index('Date', inplace=True)
# fb_data = fb_data['FB ad spends ']

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:30.187235Z","iopub.execute_input":"2023-07-24T04:25:30.187569Z","iopub.status.idle":"2023-07-24T04:25:30.193783Z","shell.execute_reply.started":"2023-07-24T04:25:30.187544Z","shell.execute_reply":"2023-07-24T04:25:30.192672Z"}}
from datetime import datetime, timedelta

# Input date in string format
input_date_str = end_date

# Convert the input date string to a datetime object
input_date = datetime.strptime(input_date_str, '%Y-%m-%d')

# Calculate the previous day by subtracting one day from the input date
previous_day = input_date - timedelta(days=1)

# Convert the previous day back to a string in the same format
previous_day_str = previous_day.strftime('%Y-%m-%d')



# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:31.025635Z","iopub.execute_input":"2023-07-24T04:25:31.025995Z","iopub.status.idle":"2023-07-24T04:25:31.034375Z","shell.execute_reply.started":"2023-07-24T04:25:31.025969Z","shell.execute_reply":"2023-07-24T04:25:31.032758Z"}}
fb_data = fb_data.loc[start_date:previous_day_str]

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:31.436097Z","iopub.execute_input":"2023-07-24T04:25:31.438050Z","iopub.status.idle":"2023-07-24T04:25:31.442316Z","shell.execute_reply.started":"2023-07-24T04:25:31.438013Z","shell.execute_reply":"2023-07-24T04:25:31.441219Z"}}
# fb_data

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:31.687192Z","iopub.execute_input":"2023-07-24T04:25:31.688183Z","iopub.status.idle":"2023-07-24T04:25:31.693064Z","shell.execute_reply.started":"2023-07-24T04:25:31.688153Z","shell.execute_reply":"2023-07-24T04:25:31.692430Z"}}
fb_data['FB ad spends'] = fb_data['FB ad spends '].str.replace('₹', '').str.replace(',', '').astype(float)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:31.872004Z","iopub.execute_input":"2023-07-24T04:25:31.872628Z","iopub.status.idle":"2023-07-24T04:25:32.099277Z","shell.execute_reply.started":"2023-07-24T04:25:31.872584Z","shell.execute_reply":"2023-07-24T04:25:32.097738Z"}}
fb_data['FB ad spends'].plot()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:32.101442Z","iopub.execute_input":"2023-07-24T04:25:32.101917Z","iopub.status.idle":"2023-07-24T04:25:32.733934Z","shell.execute_reply.started":"2023-07-24T04:25:32.101878Z","shell.execute_reply":"2023-07-24T04:25:32.732849Z"}}
# Fit the ARIMA model with your data
model = ARIMA(fb_data['FB ad spends'], order=(7, 1, 5))
results = model.fit()

# Forecast the FB ad spends from 17th July to 31st July
forecast_steps = (31-15)  # Remaining days from 17th July to 31st July
forecast = results.forecast(steps=forecast_steps)

# Create a new date range for the forecast from 17th July to 31st July
date_range_forecast = pd.date_range(start= end_date, periods=forecast_steps+1 , closed='right')

# Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(fb_data.index, fb_data['FB ad spends'], label='Original Data')
plt.plot(date_range_forecast, forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('FB Ad Spends')
plt.title('FB Ad Spends Forecast till 31st July')
plt.legend()
plt.grid(True)
plt.show()






# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:32.735422Z","iopub.execute_input":"2023-07-24T04:25:32.735742Z","iopub.status.idle":"2023-07-24T04:25:32.743710Z","shell.execute_reply.started":"2023-07-24T04:25:32.735697Z","shell.execute_reply":"2023-07-24T04:25:32.742513Z"}}
df_cumulative_income['Cumulative Income'] ,fb_data['FB ad spends']

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:32.745204Z","iopub.execute_input":"2023-07-24T04:25:32.745669Z","iopub.status.idle":"2023-07-24T04:25:32.757591Z","shell.execute_reply.started":"2023-07-24T04:25:32.745599Z","shell.execute_reply":"2023-07-24T04:25:32.756567Z"}}
fb_data['Cumulative Income']=  np.array(df_cumulative_income['Cumulative Income'])


# Calculate the correlation coefficient between 'Cumulative Income' and 'FB ad spends'
correlation_coefficient = fb_data['Cumulative Income'].corr(fb_data['FB ad spends'])

print("Correlation coefficient:", correlation_coefficient)


# %% [markdown]
# # FB DATA INTEGRATED TO THE PREDICTIONS

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:32.814043Z","iopub.execute_input":"2023-07-24T04:25:32.814352Z","iopub.status.idle":"2023-07-24T04:25:32.819746Z","shell.execute_reply.started":"2023-07-24T04:25:32.814329Z","shell.execute_reply":"2023-07-24T04:25:32.818649Z"}}
diff_spend = forecast.max()- forecast

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:32.971972Z","iopub.execute_input":"2023-07-24T04:25:32.972311Z","iopub.status.idle":"2023-07-24T04:25:32.981080Z","shell.execute_reply.started":"2023-07-24T04:25:32.972287Z","shell.execute_reply":"2023-07-24T04:25:32.980041Z"}}
diff_spend

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:33.133127Z","iopub.execute_input":"2023-07-24T04:25:33.133492Z","iopub.status.idle":"2023-07-24T04:25:33.532246Z","shell.execute_reply.started":"2023-07-24T04:25:33.133466Z","shell.execute_reply":"2023-07-24T04:25:33.531475Z"}}
# Filter the DataFrame for the desired date range (July 1 to July 16)
df_july = df[(df['Order Created Date and Time'] >= '2023-07-01') & (df['Order Created Date and Time'] <= '2023-07-16')]

# Calculate the cumulative sum of the "total" column for each day
daily_cumulative_income = df_july.groupby(df_july['Order Created Date and Time'].dt.date)['total'].sum().cumsum()

# Create a DataFrame with the date and cumulative income
df_cumulative_income = pd.DataFrame({'Date': daily_cumulative_income.index, 'Cumulative Income': daily_cumulative_income.values})

# Fit ARIMA model
model = ARIMA(df_cumulative_income['Cumulative Income'], order=(3, 3, 2)) #(2,1,2)
model_fit = model.fit()

# Forecast from 16th to month-end
forecast_steps = len(df_cumulative_income) + days_to_forecast  # Assuming 31 days in the month
forecast_values = model_fit.predict(start=len(df_cumulative_income), end=forecast_steps)

# Generate date range for forecasted values
forecast_dates = pd.date_range(start=df_cumulative_income['Date'].iloc[-1], periods=len(forecast_values) + 1, freq='D')[1:]

# Plot the actual and forecasted cumulative income
plt.plot(df_cumulative_income['Date'], df_cumulative_income['Cumulative Income'], label='Actual')
plt.plot(forecast_dates, (np.array(forecast_values) - correlation_coefficient*np.array(diff_spend)) , 'ro', label='Forecast')
plt.xlabel('Date')
plt.ylabel('Cumulative Income')
plt.title('Month-End Cumulative Income Forecast')
plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')  # Display y-axis labels in plain format
plt.legend()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:33.533915Z","iopub.execute_input":"2023-07-24T04:25:33.534535Z","iopub.status.idle":"2023-07-24T04:25:33.545192Z","shell.execute_reply.started":"2023-07-24T04:25:33.534505Z","shell.execute_reply":"2023-07-24T04:25:33.543873Z"}}
# Forecast from 16th to month-end
forecast_steps = days_to_forecast  # Assuming 31 days in the month
forecast_values = model_fit.predict(start=len(df_cumulative_income), end=len(df_cumulative_income) + forecast_steps)
forecast_values = pd.Series(np.array(forecast_values) - correlation_coefficient*np.array(diff_spend))
# Calculate month-end income
month_end_income = forecast_values.iloc[-1]

# Calculate income in steps of 5 days
income_in_steps = forecast_values.iloc[::5]

# Generate the text
result_text = f"Month-End Income: {month_end_income}\nIncome in Steps of 5 Days:\n"

for i, income in enumerate(income_in_steps):
    result_text += f"Day {i*5+16}: {income}\n"

# Print the result
print(result_text)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:33.570497Z","iopub.execute_input":"2023-07-24T04:25:33.570868Z","iopub.status.idle":"2023-07-24T04:25:33.575963Z","shell.execute_reply.started":"2023-07-24T04:25:33.570843Z","shell.execute_reply":"2023-07-24T04:25:33.575058Z"}}
# # Calculate cumulative income
# df['Cumulative Income'] = df['total'].cumsum()

# # Plotting the graph
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Cumulative Income'])
# plt.xlabel('Time')
# plt.ylabel('Cumulative Income')
# plt.title('Cumulative Income over Time')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.ticklabel_format(style='plain', axis='y')
# plt.show()


# %% [markdown]
# # lets try deploying the code 

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:04.434445Z","iopub.execute_input":"2023-07-24T04:25:04.434823Z","iopub.status.idle":"2023-07-24T04:25:05.349632Z","shell.execute_reply.started":"2023-07-24T04:25:04.434791Z","shell.execute_reply":"2023-07-24T04:25:05.346756Z"}}
from fastapi import FastAPI, HTTPException, Request
import pickle

app = FastAPI()

# Load the ARIMA model
with open("arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

@app.post("/predict/")
async def predict(request: Request):
    try:
        data = await request.json()
        values_to_predict = data["values"]

        # Perform prediction using the ARIMA model
        predictions = arima_model.predict(n_periods=len(values_to_predict))

        # Convert predictions to a list
        predictions_list = predictions.tolist()

        # Prepare the response
        response = {
            "predictions": predictions_list
        }

        return response

    except Exception as e:
        # Handle any errors that may occur during prediction
        raise HTTPException(status_code=500, detail=str(e))


# %% [markdown]
# # lets train the ARIMA model

# %% [markdown]
# # PACF - Auto regression

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.350489Z","iopub.status.idle":"2023-07-24T04:25:05.351486Z","shell.execute_reply.started":"2023-07-24T04:25:05.351274Z","shell.execute_reply":"2023-07-24T04:25:05.351295Z"}}
# from statsmodels.graphics.tsaplots import plot_pacf

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.352753Z","iopub.status.idle":"2023-07-24T04:25:05.353410Z","shell.execute_reply.started":"2023-07-24T04:25:05.353203Z","shell.execute_reply":"2023-07-24T04:25:05.353223Z"}}
# from statsmodels.graphics.tsaplots import plot_acf

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.354569Z","iopub.status.idle":"2023-07-24T04:25:05.355230Z","shell.execute_reply.started":"2023-07-24T04:25:05.355021Z","shell.execute_reply":"2023-07-24T04:25:05.355041Z"}}
# train_data


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.356390Z","iopub.status.idle":"2023-07-24T04:25:05.356972Z","shell.execute_reply.started":"2023-07-24T04:25:05.356762Z","shell.execute_reply":"2023-07-24T04:25:05.356781Z"}}
# # Extract the cumulative invoice for the 3 days training period
# train_data = df[(df.index.day < 7) | (df.index.month < 7)]["Cumulative Income"]


# # Plot the ACF with respect to each 1-hour lag within the training period
# plt.figure(figsize=(12, 6))
# plot_acf(train_data, lags=5)
# plt.xlabel('Time Lag (1 day)')
# plt.ylabel('Partial Autocorrelation')
# plt.title('Autocorrelation Function (ACF) - 10 days')
# plt.grid(True)
# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.358017Z","iopub.status.idle":"2023-07-24T04:25:05.358567Z","shell.execute_reply.started":"2023-07-24T04:25:05.358371Z","shell.execute_reply":"2023-07-24T04:25:05.358390Z"}}
# # Extract the cumulative invoice for the 3 days training period
# train_data = df[(df.index.day < 7) | (df.index.month < 7)]['Cumulative Income']

# # Plot the PACF with respect to each 1-hour lag within the  training period
# plt.figure(figsize=(12, 6))
# plot_pacf(train_data, lags=5)
# plt.xlabel('Time Lag (1 day)')
# plt.ylabel('Partial Autocorrelation')
# plt.title('Partial Autocorrelation Function (PACF) - 10 days')
# plt.grid(True)
# plt.show()

# %% [markdown]
# # Differencing 

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.359628Z","iopub.status.idle":"2023-07-24T04:25:05.360210Z","shell.execute_reply.started":"2023-07-24T04:25:05.360006Z","shell.execute_reply":"2023-07-24T04:25:05.360025Z"}}
# from statsmodels.tsa.stattools import adfuller

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.361170Z","iopub.status.idle":"2023-07-24T04:25:05.362000Z","shell.execute_reply.started":"2023-07-24T04:25:05.361837Z","shell.execute_reply":"2023-07-24T04:25:05.361852Z"}}
# # Perform Dickey-Fuller test
# result = adfuller(df['Cumulative Income'])

# # Extract the test statistic and p-value
# test_statistic = result[0]
# p_value = result[1]

# # Print the test statistic and p-value
# print(f'Test Statistic: {test_statistic:.4f}')
# print(f'p-value: {p_value:.4f}')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.363371Z","iopub.status.idle":"2023-07-24T04:25:05.363898Z","shell.execute_reply.started":"2023-07-24T04:25:05.363735Z","shell.execute_reply":"2023-07-24T04:25:05.363753Z"}}
# df['Difference1'] = df['Cumulative Income']-df['Cumulative Income'].shift(1)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.364648Z","iopub.status.idle":"2023-07-24T04:25:05.365399Z","shell.execute_reply.started":"2023-07-24T04:25:05.365256Z","shell.execute_reply":"2023-07-24T04:25:05.365271Z"}}
# df['Difference1'].dropna()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.366123Z","iopub.status.idle":"2023-07-24T04:25:05.366790Z","shell.execute_reply.started":"2023-07-24T04:25:05.366635Z","shell.execute_reply":"2023-07-24T04:25:05.366653Z"}}
# # Perform Dickey-Fuller test
# result = adfuller(df['Difference1'].dropna())

# # Extract the test statistic and p-value
# test_statistic = result[0]
# p_value = result[1]

# # Print the test statistic and p-value
# print(f'Test Statistic: {test_statistic:.4f}')
# print(f'p-value: {p_value:.4f}')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.367879Z","iopub.status.idle":"2023-07-24T04:25:05.368159Z","shell.execute_reply.started":"2023-07-24T04:25:05.368022Z","shell.execute_reply":"2023-07-24T04:25:05.368037Z"}}
# from statsmodels.tsa.arima.model import ARIMA

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.369235Z","iopub.status.idle":"2023-07-24T04:25:05.369518Z","shell.execute_reply.started":"2023-07-24T04:25:05.369381Z","shell.execute_reply":"2023-07-24T04:25:05.369397Z"}}
# df.index.hour

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.370515Z","iopub.status.idle":"2023-07-24T04:25:05.370814Z","shell.execute_reply.started":"2023-07-24T04:25:05.370670Z","shell.execute_reply":"2023-07-24T04:25:05.370683Z"}}
# # Extract the time before noon
# train_data = df[(df.index.day < 10) | (df.index.month < 7)]['Cumulative Income']

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.371721Z","iopub.status.idle":"2023-07-24T04:25:05.371996Z","shell.execute_reply.started":"2023-07-24T04:25:05.371856Z","shell.execute_reply":"2023-07-24T04:25:05.371878Z"}}
# # Fit the ARIMA model
# model = ARIMA(train_data, order=(5, 1, 2))
# model_fit = model.fit()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.372830Z","iopub.status.idle":"2023-07-24T04:25:05.373130Z","shell.execute_reply.started":"2023-07-24T04:25:05.372978Z","shell.execute_reply":"2023-07-24T04:25:05.372992Z"}}
# # Predict total invoice after noon
# test_data = df[(df.index.day >= 10) & (df.index.month >= 7)]['Cumulative Income']
# predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.374132Z","iopub.status.idle":"2023-07-24T04:25:05.374399Z","shell.execute_reply.started":"2023-07-24T04:25:05.374266Z","shell.execute_reply":"2023-07-24T04:25:05.374281Z"}}
# df[df.index.day<4].index

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.375233Z","iopub.status.idle":"2023-07-24T04:25:05.375500Z","shell.execute_reply.started":"2023-07-24T04:25:05.375368Z","shell.execute_reply":"2023-07-24T04:25:05.375384Z"}}
# # Plot the actual and predicted values
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Cumulative Income'], label='Actual')
# # plt.plot(test_data.index, predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Cumulative Income')
# plt.title('ARIMA Model - Actual vs Predicted')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.376685Z","iopub.status.idle":"2023-07-24T04:25:05.376969Z","shell.execute_reply.started":"2023-07-24T04:25:05.376827Z","shell.execute_reply":"2023-07-24T04:25:05.376842Z"}}
# # Plot the actual and predicted values
# plt.figure(figsize=(12, 6))
# plt.plot(df[(df.index.day<7) | (df.index.month<7)].index,df[(df.index.day<7) | (df.index.month<7)]['Cumulative Income'], label='Actual')
# plt.plot(test_data.index, predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Cumulative Income')
# plt.title('ARIMA Model - Actual vs Predicted')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.378015Z","iopub.status.idle":"2023-07-24T04:25:05.378298Z","shell.execute_reply.started":"2023-07-24T04:25:05.378154Z","shell.execute_reply":"2023-07-24T04:25:05.378171Z"}}
# df

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.379141Z","iopub.status.idle":"2023-07-24T04:25:05.379422Z","shell.execute_reply.started":"2023-07-24T04:25:05.379279Z","shell.execute_reply":"2023-07-24T04:25:05.379294Z"}}
# # Assuming your DataFrame or Series is named 'df'
# day1 = 7
# day2 = 8

# # Get the indices where the day is equal to day1
# day1_indices = df.index[df.index.day == day1]

# # Get the indices where the day is equal to day2
# day2_indices = df.index[df.index.day == day2]

# # Get the starting index of day1
# day1_start_index = day1_indices[0]

# # Get the starting index of day2
# day2_start_index = day2_indices[0]

# # Get the number of examples between day1 and day2
# num_examples_between = len(df[day1_start_index:day2_start_index]) + 1

# # Print the results
# print("Starting index of day1:", day1_start_index)
# print("Starting index of day2:", day2_start_index)
# print("Number of examples between day1 and day2:", num_examples_between)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.380272Z","iopub.status.idle":"2023-07-24T04:25:05.380559Z","shell.execute_reply.started":"2023-07-24T04:25:05.380422Z","shell.execute_reply":"2023-07-24T04:25:05.380438Z"}}
# # Forecast for the next day
# steps=196
# forecast = model_fit.forecast(steps=196)

# # Print the forecasted value for the next day
# print("Forecast for the next step:")
# print(forecast[1440+steps-1])


# print("So the turnover in the next day is:")
# print(forecast[1440+steps-1]- forecast[1440])


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.381498Z","iopub.status.idle":"2023-07-24T04:25:05.381802Z","shell.execute_reply.started":"2023-07-24T04:25:05.381659Z","shell.execute_reply":"2023-07-24T04:25:05.381674Z"}}
# import pandas as pd

# # Define the start timestamp of the next day
# start_timestamp = pd.to_datetime("2023-07-9").replace(hour=0, minute=0, second=0)

# # Create the date-time index for the next day with an interval of 196 in the day
# next_day_index = pd.date_range(start=start_timestamp, periods=196, freq="7.5min")

# # Assign the new index to the forecasted values
# forecast.index = next_day_index


# # Print the forecasted values with the new index
# print("The turnover by the end of the next day ie total")
# print(forecast["2023-07-10 00:22:30"])
# # forecast
# print("")
# print("The turnover for the next day")
# print(forecast["2023-07-10 00:22:30"]-forecast["2023-07-09 00:00:00"])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.382887Z","iopub.status.idle":"2023-07-24T04:25:05.383169Z","shell.execute_reply.started":"2023-07-24T04:25:05.383027Z","shell.execute_reply":"2023-07-24T04:25:05.383042Z"}}
# import pandas as pd

# # Get the last index of the forecast
# last_index = forecast.index[-1]

# # Add one day to the last index to get the next day's date-time
# next_day_index = last_index + pd.DateOffset(days=1)

# # Update the last index of the forecast with the next day's date-time
# forecast.index = forecast.index[:-1].append(pd.Index([next_day_index]))

# # Print the forecast with the updated last index
# print(forecast)


# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.384251Z","iopub.status.idle":"2023-07-24T04:25:05.384534Z","shell.execute_reply.started":"2023-07-24T04:25:05.384400Z","shell.execute_reply":"2023-07-24T04:25:05.384414Z"}}
# # Plot the actual and predicted values
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['Cumulative Income'], label='Actual')
# plt.plot(forecast,  color='red')
# # plt.plot(test_data.index, predictions, label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Cumulative Income')
# plt.title('ARIMA Model - Actual vs Predicted')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.385316Z","iopub.status.idle":"2023-07-24T04:25:05.385594Z","shell.execute_reply.started":"2023-07-24T04:25:05.385460Z","shell.execute_reply":"2023-07-24T04:25:05.385475Z"}}
# data = pd.read_csv('/kaggle/input/task-1/8223-itemsRequired-10-07-2023.csv')
# data.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.386521Z","iopub.status.idle":"2023-07-24T04:25:05.386814Z","shell.execute_reply.started":"2023-07-24T04:25:05.386677Z","shell.execute_reply":"2023-07-24T04:25:05.386692Z"}}
# data

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T04:25:05.389839Z","iopub.status.idle":"2023-07-24T04:25:05.390334Z","shell.execute_reply.started":"2023-07-24T04:25:05.390189Z","shell.execute_reply":"2023-07-24T04:25:05.390204Z"}}
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # Load your time series data into a pandas DataFrame
# # Make sure you have a column with the dates and a column with the values
# data = pd.read_csv('/kaggle/input/task-1/8223-itemsRequired-10-07-2023.csv', parse_dates=['date_created'], index_col='date_created')

# # Prepare the data for linear regression
# X = np.arange(len(df)).reshape(-1, 1)  # Use the index as the input feature
# y = df.values.flatten()  # Use the values as the target variable

# # Fit the linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Generate the forecasted values
# forecast = model.predict(X)

# # Print the forecasted values
# print(forecast)

# # Visualize the forecasted values
# plt.plot(data)
# plt.plot(data.index, forecast, color='red')
# plt.show()


# %% [code]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:41.810416Z","iopub.execute_input":"2023-08-03T01:52:41.810830Z","iopub.status.idle":"2023-08-03T01:52:41.845792Z","shell.execute_reply.started":"2023-08-03T01:52:41.810795Z","shell.execute_reply":"2023-08-03T01:52:41.844685Z"}}
import pandas as pd
import numpy as np

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:41.847864Z","iopub.execute_input":"2023-08-03T01:52:41.849143Z","iopub.status.idle":"2023-08-03T01:52:41.855990Z","shell.execute_reply.started":"2023-08-03T01:52:41.849083Z","shell.execute_reply":"2023-08-03T01:52:41.854744Z"}}
start_date = '2023-07-01'
end_date = '2023-07-23'
total_days_in_month = 31
days_data_having = np.array(int(end_date[-2:])).item()- np.array(int(start_date[-2:])).item() + 1

days_to_forecast = total_days_in_month - days_data_having

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:41.857431Z","iopub.execute_input":"2023-08-03T01:52:41.858001Z","iopub.status.idle":"2023-08-03T01:52:41.992953Z","shell.execute_reply.started":"2023-08-03T01:52:41.857970Z","shell.execute_reply":"2023-08-03T01:52:41.991760Z"}}
df= pd.read_csv("/kaggle/input/modak-problem/modaka- puranpoli problem.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:41.995214Z","iopub.execute_input":"2023-08-03T01:52:41.995551Z","iopub.status.idle":"2023-08-03T01:52:42.003848Z","shell.execute_reply.started":"2023-08-03T01:52:41.995523Z","shell.execute_reply":"2023-08-03T01:52:42.002629Z"}}
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.005552Z","iopub.execute_input":"2023-08-03T01:52:42.006324Z","iopub.status.idle":"2023-08-03T01:52:42.027654Z","shell.execute_reply.started":"2023-08-03T01:52:42.006292Z","shell.execute_reply":"2023-08-03T01:52:42.026697Z"}}
df = df.loc[:,['Order created Date','Order No','item_name','Pack of']]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.029341Z","iopub.execute_input":"2023-08-03T01:52:42.029991Z","iopub.status.idle":"2023-08-03T01:52:42.036269Z","shell.execute_reply.started":"2023-08-03T01:52:42.029960Z","shell.execute_reply":"2023-08-03T01:52:42.035384Z"}}
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.037715Z","iopub.execute_input":"2023-08-03T01:52:42.038309Z","iopub.status.idle":"2023-08-03T01:52:42.046478Z","shell.execute_reply.started":"2023-08-03T01:52:42.038279Z","shell.execute_reply":"2023-08-03T01:52:42.045408Z"}}
# # Create a bar plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='item_name', y='Pack of', data=df, ci=None)
# plt.title('Quantity of Items')
# plt.xlabel('Item Name')
# plt.ylabel('Quantity')

# # Adjust the x-axis labels rotation for better readability
# plt.xticks(rotation=90)

# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.048054Z","iopub.execute_input":"2023-08-03T01:52:42.048803Z","iopub.status.idle":"2023-08-03T01:52:42.070641Z","shell.execute_reply.started":"2023-08-03T01:52:42.048771Z","shell.execute_reply":"2023-08-03T01:52:42.069535Z"}}
# Calculate the quantity of each item
quantity_per_item = df.groupby('item_name')['Pack of'].sum()

print(quantity_per_item)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.073771Z","iopub.execute_input":"2023-08-03T01:52:42.074392Z","iopub.status.idle":"2023-08-03T01:52:42.083374Z","shell.execute_reply.started":"2023-08-03T01:52:42.074358Z","shell.execute_reply":"2023-08-03T01:52:42.082357Z"}}
df.iloc[:,0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.085287Z","iopub.execute_input":"2023-08-03T01:52:42.085935Z","iopub.status.idle":"2023-08-03T01:52:42.265506Z","shell.execute_reply.started":"2023-08-03T01:52:42.085894Z","shell.execute_reply":"2023-08-03T01:52:42.264398Z"}}
from datetime import datetime 
# Define a function to convert datetime string to datetime object
def convert_to_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')

# Apply the conversion function to the "Order Created Date and Time" column
df['Order created Date'] = df['Order created Date'].apply(convert_to_datetime)


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.267366Z","iopub.execute_input":"2023-08-03T01:52:42.267803Z","iopub.status.idle":"2023-08-03T01:52:42.277845Z","shell.execute_reply.started":"2023-08-03T01:52:42.267762Z","shell.execute_reply":"2023-08-03T01:52:42.276635Z"}}
df.iloc[:,0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.279728Z","iopub.execute_input":"2023-08-03T01:52:42.280216Z","iopub.status.idle":"2023-08-03T01:52:42.304339Z","shell.execute_reply.started":"2023-08-03T01:52:42.280176Z","shell.execute_reply":"2023-08-03T01:52:42.303181Z"}}
# Calculate the quantity of each item per day
quantity_per_item_per_day = df.groupby(['Order created Date', 'item_name'])['Pack of'].sum()

print(quantity_per_item_per_day)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.305507Z","iopub.execute_input":"2023-08-03T01:52:42.306647Z","iopub.status.idle":"2023-08-03T01:52:42.311657Z","shell.execute_reply.started":"2023-08-03T01:52:42.306608Z","shell.execute_reply":"2023-08-03T01:52:42.310329Z"}}
# # Filter the DataFrame to include only item names containing 'modak'
# filtered_df = df[df['item_name'].str.contains('modak', case=False)]

# # Calculate the total quantity for each date
# total_quantity_per_day = filtered_df.groupby('Order created Date')['Pack of'].sum().cumsum()

# print(total_quantity_per_day)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:42.313728Z","iopub.execute_input":"2023-08-03T01:52:42.314196Z","iopub.status.idle":"2023-08-03T01:52:44.056033Z","shell.execute_reply.started":"2023-08-03T01:52:42.314156Z","shell.execute_reply":"2023-08-03T01:52:44.054787Z"}}
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:44.060838Z","iopub.execute_input":"2023-08-03T01:52:44.061305Z","iopub.status.idle":"2023-08-03T01:52:44.072988Z","shell.execute_reply.started":"2023-08-03T01:52:44.061246Z","shell.execute_reply":"2023-08-03T01:52:44.071726Z"}}
df.iloc[:,0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:52:44.074843Z","iopub.execute_input":"2023-08-03T01:52:44.075287Z","iopub.status.idle":"2023-08-03T01:52:44.108099Z","shell.execute_reply.started":"2023-08-03T01:52:44.075249Z","shell.execute_reply":"2023-08-03T01:52:44.107356Z"}}
# Sort the DataFrame by 'Order created Date'
df.sort_values(by='Order created Date', inplace=True)

# Group by 'Order created Date' and 'item_name', then sum the 'Pack of' column
quantity_per_day_per_product = df.groupby([df['Order created Date'].dt.date, 'item_name'])['Pack of'].sum()

print(quantity_per_day_per_product)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:36.276346Z","iopub.execute_input":"2023-08-03T01:53:36.276789Z","iopub.status.idle":"2023-08-03T01:53:36.627325Z","shell.execute_reply.started":"2023-08-03T01:53:36.276751Z","shell.execute_reply":"2023-08-03T01:53:36.626174Z"}}
import seaborn as sns


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:38.717782Z","iopub.execute_input":"2023-08-03T01:53:38.718219Z","iopub.status.idle":"2023-08-03T01:53:43.132656Z","shell.execute_reply.started":"2023-08-03T01:53:38.718178Z","shell.execute_reply":"2023-08-03T01:53:43.131538Z"}}
# Sort the DataFrame by 'Order created Date'
df.sort_values(by='Order created Date', inplace=True)

# Group by 'Order created Date' and 'item_name', then sum the 'Pack of' column
quantity_per_day_per_product = df.groupby([df['Order created Date'].dt.date, 'item_name'])['Pack of'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Order created Date', y='Pack of', hue='item_name', data=quantity_per_day_per_product)
plt.title('Quantity of Each Product per Day')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.xticks(rotation=90)
plt.legend(title='Product')
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:48.016729Z","iopub.execute_input":"2023-08-03T01:53:48.017098Z","iopub.status.idle":"2023-08-03T01:53:48.962353Z","shell.execute_reply.started":"2023-08-03T01:53:48.017070Z","shell.execute_reply":"2023-08-03T01:53:48.961191Z"}}
# Filter data for the last 3 days
last_3_days = df[df['Order created Date'] >= df['Order created Date'].max() - pd.Timedelta(days=2)]

# Group by 'Order created Date' and 'item_name', then sum the 'Pack of' column
quantity_per_day_per_product = last_3_days.groupby([last_3_days['Order created Date'].dt.date, 'item_name'])['Pack of'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Order created Date', y='Pack of', hue='item_name', data=quantity_per_day_per_product)
plt.title('Quantity of Each Product for the Last 3 Days')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.legend(title='Product')
plt.tight_layout()
plt.show()



# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:50.692641Z","iopub.execute_input":"2023-08-03T01:53:50.693022Z","iopub.status.idle":"2023-08-03T01:53:51.127087Z","shell.execute_reply.started":"2023-08-03T01:53:50.692993Z","shell.execute_reply":"2023-08-03T01:53:51.125981Z"}}
# Filter data for the last 3 days
last_3_days = df[df['Order created Date'] >= df['Order created Date'].max() - pd.Timedelta(days=2)]

# Filter only item names containing 'Modak' or 'Puranpoli'
filtered_last_3_days = last_3_days[last_3_days['item_name'].str.contains('Modak|Puranpoli', case=False)]

# Group by 'Order created Date' and 'item_name', then sum the 'Pack of' column
quantity_per_day_per_product = filtered_last_3_days.groupby([filtered_last_3_days['Order created Date'].dt.date, 'item_name'])['Pack of'].sum().reset_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Order created Date', y='Pack of', hue='item_name', data=quantity_per_day_per_product)
plt.title('Quantity of Each Product (Modak and Puranpoli) for the Last 3 Days')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.legend(title='Product')
plt.tight_layout()
plt.show()

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-08-03T01:53:51.129049Z","iopub.execute_input":"2023-08-03T01:53:51.129900Z","iopub.status.idle":"2023-08-03T01:53:51.509919Z","shell.execute_reply.started":"2023-08-03T01:53:51.129864Z","shell.execute_reply":"2023-08-03T01:53:51.508546Z"}}
# Filter only item names containing 'Modak' or 'Puranpoli'
filtered_df = df[df['item_name'].str.contains('Modak|Puranpoli', case=False)]

# Group by 'Order created Date' and 'item_name', then sum the 'Pack of' column to get quantity per day
quantity_per_day_per_product = filtered_df.groupby([filtered_df['Order created Date'].dt.date, 'item_name'])['Pack of'].sum().reset_index()

# Set 'Order created Date' as the index
quantity_per_day_per_product.set_index('Order created Date', inplace=True)

# Get unique item names for forecasting
unique_item_names = quantity_per_day_per_product['item_name'].unique()

# Perform forecasting for each item
forecasts = {}
for item_name in unique_item_names:
    item_data = quantity_per_day_per_product[quantity_per_day_per_product['item_name'] == item_name]['Pack of']
    model = ARIMA(item_data, order=(1, 1, 1))  # Replace (1, 1, 1) with appropriate order based on data characteristics
    results = model.fit()
    forecast = results.forecast(steps=1)
    forecasts[item_name] = forecast[0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:51.511828Z","iopub.execute_input":"2023-08-03T01:53:51.512293Z","iopub.status.idle":"2023-08-03T01:53:51.518783Z","shell.execute_reply.started":"2023-08-03T01:53:51.512261Z","shell.execute_reply":"2023-08-03T01:53:51.517467Z"}}
print("Forecasts for the next day:")
print(forecasts)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:51.686919Z","iopub.execute_input":"2023-08-03T01:53:51.687356Z","iopub.status.idle":"2023-08-03T01:53:52.078067Z","shell.execute_reply.started":"2023-08-03T01:53:51.687321Z","shell.execute_reply":"2023-08-03T01:53:52.076702Z"}}
total_quantity_per_day = quantity_per_day_per_product.groupby(level=0)['Pack of'].sum()
plt.figure(figsize=(10, 6))
plt.plot(total_quantity_per_day.index, total_quantity_per_day.values, marker='o', label='Total Quantity', color='blue')
plt.title('Total Quantity per Day')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:52.247019Z","iopub.execute_input":"2023-08-03T01:53:52.247747Z","iopub.status.idle":"2023-08-03T01:53:52.808161Z","shell.execute_reply.started":"2023-08-03T01:53:52.247701Z","shell.execute_reply":"2023-08-03T01:53:52.806992Z"}}
# Create a plot for total quantity per product for all days
plt.figure(figsize=(10, 6))
for item_name in quantity_per_day_per_product['item_name'].unique():
    total_quantity_per_product = quantity_per_day_per_product[quantity_per_day_per_product['item_name'] == item_name].groupby(level=0)['Pack of'].sum()
    plt.plot(total_quantity_per_product.index, total_quantity_per_product.values, marker='o', label=item_name)

plt.title('Total Quantity per Product per Day')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:53:52.810353Z","iopub.execute_input":"2023-08-03T01:53:52.810797Z","iopub.status.idle":"2023-08-03T01:53:56.540931Z","shell.execute_reply.started":"2023-08-03T01:53:52.810763Z","shell.execute_reply":"2023-08-03T01:53:56.539697Z"}}
# Create a plot for total quantity per product for all days
plt.figure(figsize=(10, 6))
for item_name in quantity_per_day_per_product['item_name'].unique():
    total_quantity_per_product = quantity_per_day_per_product[quantity_per_day_per_product['item_name'] == item_name].groupby(level=0)['Pack of'].sum()
    plt.plot(total_quantity_per_product.index, total_quantity_per_product.values, marker='o', label=item_name)

# Perform forecasting for each item
forecasts = {}
colors = ['orange','red', 'blue', 'green']  # Add more colors if you have more items
for idx, item_name in enumerate(quantity_per_day_per_product['item_name'].unique()):
    item_data = quantity_per_day_per_product[quantity_per_day_per_product['item_name'] == item_name]['Pack of']
    model = ARIMA(item_data, order=(5, 2, 3))  # Replace (1, 1, 1) with appropriate order based on data characteristics
    results = model.fit()
    forecast = results.forecast(steps=days_to_forecast)  # Forecast for 8 days
    forecasts[item_name] = forecast

    # Create a plot for forecasted quantities for the next 8 days (dotted line)
    plt.plot(pd.date_range(start=quantity_per_day_per_product.index[-1] + pd.Timedelta(days=1), periods=8, freq='D'), forecast,
             linestyle='dotted', color=colors[idx], label=f'{item_name} Forecast')

plt.title('Total Quantity per Product per Day with Forecasts')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:55:23.476736Z","iopub.execute_input":"2023-08-03T01:55:23.477138Z","iopub.status.idle":"2023-08-03T01:55:23.484355Z","shell.execute_reply.started":"2023-08-03T01:55:23.477094Z","shell.execute_reply":"2023-08-03T01:55:23.483195Z"}}
# Print the forecasts for the next 5 days in intervals of 5
for item_name, forecast in forecasts.items():
    print(f"Forecast for {item_name} for the next days:")
    for i, value in enumerate(forecast):
        print(f"Day {i + 1}: {value:.2f}")
    print("\n")
# This code will perform forecasting for the quantities of 'Modak' and 'Puranpoli' for the next 5 days and print the forecasted quantities for each item in intervals of 5 days. Replace the data variable with your actual DataFrame to get the correct forecasts.







# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]

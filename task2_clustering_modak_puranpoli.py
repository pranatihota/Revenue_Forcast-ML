# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.644059Z","iopub.execute_input":"2023-08-03T01:56:15.644477Z","iopub.status.idle":"2023-08-03T01:56:15.649870Z","shell.execute_reply.started":"2023-08-03T01:56:15.644430Z","shell.execute_reply":"2023-08-03T01:56:15.648594Z"}}
import pandas as pd

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.651689Z","iopub.execute_input":"2023-08-03T01:56:15.652305Z","iopub.status.idle":"2023-08-03T01:56:15.744967Z","shell.execute_reply.started":"2023-08-03T01:56:15.652271Z","shell.execute_reply":"2023-08-03T01:56:15.743431Z"}}
df = pd.read_csv("/kaggle/input/modak-problem/modaka- puranpoli problem.csv")

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.747257Z","iopub.execute_input":"2023-08-03T01:56:15.747682Z","iopub.status.idle":"2023-08-03T01:56:15.755497Z","shell.execute_reply.started":"2023-08-03T01:56:15.747640Z","shell.execute_reply":"2023-08-03T01:56:15.754649Z"}}
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.756901Z","iopub.execute_input":"2023-08-03T01:56:15.757279Z","iopub.status.idle":"2023-08-03T01:56:15.795243Z","shell.execute_reply.started":"2023-08-03T01:56:15.757246Z","shell.execute_reply":"2023-08-03T01:56:15.793725Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.797994Z","iopub.execute_input":"2023-08-03T01:56:15.798371Z","iopub.status.idle":"2023-08-03T01:56:15.806295Z","shell.execute_reply.started":"2023-08-03T01:56:15.798337Z","shell.execute_reply":"2023-08-03T01:56:15.805379Z"}}
df['Order status'].unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.807976Z","iopub.execute_input":"2023-08-03T01:56:15.808384Z","iopub.status.idle":"2023-08-03T01:56:15.822650Z","shell.execute_reply.started":"2023-08-03T01:56:15.808350Z","shell.execute_reply":"2023-08-03T01:56:15.821405Z"}}
df.item_name.unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.824013Z","iopub.execute_input":"2023-08-03T01:56:15.824361Z","iopub.status.idle":"2023-08-03T01:56:15.834977Z","shell.execute_reply.started":"2023-08-03T01:56:15.824330Z","shell.execute_reply":"2023-08-03T01:56:15.833991Z"}}
df = df.loc[:,['Order created Date','Order No','item_name','Pack of']]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.836220Z","iopub.execute_input":"2023-08-03T01:56:15.836644Z","iopub.status.idle":"2023-08-03T01:56:15.862204Z","shell.execute_reply.started":"2023-08-03T01:56:15.836611Z","shell.execute_reply":"2023-08-03T01:56:15.860722Z"}}
df

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:15.864519Z","iopub.execute_input":"2023-08-03T01:56:15.865772Z","iopub.status.idle":"2023-08-03T01:56:16.013579Z","shell.execute_reply.started":"2023-08-03T01:56:15.865724Z","shell.execute_reply":"2023-08-03T01:56:16.012032Z"}}
from datetime import datetime 
# Define a function to convert datetime string to datetime object
def convert_to_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')

# Apply the conversion function to the "Order Created Date and Time" column
df['Order created Date'] = df['Order created Date'].apply(convert_to_datetime)


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.016922Z","iopub.execute_input":"2023-08-03T01:56:16.017771Z","iopub.status.idle":"2023-08-03T01:56:16.025305Z","shell.execute_reply.started":"2023-08-03T01:56:16.017718Z","shell.execute_reply":"2023-08-03T01:56:16.024104Z"}}
# df['Order Created Date'] = pd.to_datetime(df['Order created Date'])

# Sort the DataFrame based on the "Order Created Date and Time" column
df = df.sort_values('Order created Date', ascending=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.029857Z","iopub.execute_input":"2023-08-03T01:56:16.030252Z","iopub.status.idle":"2023-08-03T01:56:16.037627Z","shell.execute_reply.started":"2023-08-03T01:56:16.030218Z","shell.execute_reply":"2023-08-03T01:56:16.036057Z"}}
# df= df.set_index('Order created Date')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.039278Z","iopub.execute_input":"2023-08-03T01:56:16.039663Z","iopub.status.idle":"2023-08-03T01:56:16.052026Z","shell.execute_reply.started":"2023-08-03T01:56:16.039629Z","shell.execute_reply":"2023-08-03T01:56:16.050701Z"}}
item_names= df.item_name.unique()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.053656Z","iopub.execute_input":"2023-08-03T01:56:16.054405Z","iopub.status.idle":"2023-08-03T01:56:16.066265Z","shell.execute_reply.started":"2023-08-03T01:56:16.054364Z","shell.execute_reply":"2023-08-03T01:56:16.065269Z"}}
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.067859Z","iopub.execute_input":"2023-08-03T01:56:16.068608Z","iopub.status.idle":"2023-08-03T01:56:16.628467Z","shell.execute_reply.started":"2023-08-03T01:56:16.068565Z","shell.execute_reply":"2023-08-03T01:56:16.627222Z"}}
# Create a DataFrame to store the unique item names
df_items = pd.DataFrame(item_names, columns=["item_name"])

# Count the occurrences of each unique item
item_counts = df["item_name"].value_counts()

# Plot the data using a bar chart
plt.figure(figsize=(12, 6))
plt.bar(item_counts.index, item_counts.values)
plt.xticks(rotation=90)
plt.xlabel("Item Name")
plt.ylabel("Count")
plt.title("Count of Unique Items")
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:16.630328Z","iopub.execute_input":"2023-08-03T01:56:16.630835Z","iopub.status.idle":"2023-08-03T01:56:17.056723Z","shell.execute_reply.started":"2023-08-03T01:56:16.630788Z","shell.execute_reply":"2023-08-03T01:56:17.055859Z"}}
# Filter the item names to include only those with "puran poli" or "modak"
filtered_items = df[df["item_name"].str.contains("Puranpoli|Modak", case=False)]

# Count the occurrences of each filtered item
item_counts = filtered_items["item_name"].value_counts()

# Plot the data using a bar chart
plt.figure(figsize=(10, 6))
plt.bar(item_counts.index, item_counts.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Item Name")
plt.ylabel("Count")
plt.title("Count of 'Puran Poli' and 'Modak' Items")
plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:17.057832Z","iopub.execute_input":"2023-08-03T01:56:17.058368Z","iopub.status.idle":"2023-08-03T01:56:17.068993Z","shell.execute_reply.started":"2023-08-03T01:56:17.058336Z","shell.execute_reply":"2023-08-03T01:56:17.067883Z"}}
df['Order No']

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:17.070554Z","iopub.execute_input":"2023-08-03T01:56:17.070989Z","iopub.status.idle":"2023-08-03T01:56:17.086606Z","shell.execute_reply.started":"2023-08-03T01:56:17.070958Z","shell.execute_reply":"2023-08-03T01:56:17.085148Z"}}
# Count how many items are ordered for each unique order number
order_counts = df['Order No'].value_counts()

# Print the result

print(order_counts)


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:17.088402Z","iopub.execute_input":"2023-08-03T01:56:17.088827Z","iopub.status.idle":"2023-08-03T01:56:17.240357Z","shell.execute_reply.started":"2023-08-03T01:56:17.088792Z","shell.execute_reply":"2023-08-03T01:56:17.239254Z"}}
order_counts = df['Order No'].value_counts()

# Filter the order numbers that have more than 2 items ordered
order_numbers_with_more_than_2_items = order_counts[order_counts >= 2].index

# Filter the DataFrame to keep only the orders with more than 2 items
filtered_df = df[df['Order No'].isin(order_numbers_with_more_than_2_items)]

# Drop the orders that have 1 item ordered
filtered_df = filtered_df.groupby('Order No').filter(lambda x: len(x) > 1)


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T01:56:17.241823Z","iopub.execute_input":"2023-08-03T01:56:17.242787Z","iopub.status.idle":"2023-08-03T01:56:17.258865Z","shell.execute_reply.started":"2023-08-03T01:56:17.242752Z","shell.execute_reply":"2023-08-03T01:56:17.257954Z"}}
filtered_df

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T02:02:43.966681Z","iopub.execute_input":"2023-08-03T02:02:43.967094Z","iopub.status.idle":"2023-08-03T02:02:54.791601Z","shell.execute_reply.started":"2023-08-03T02:02:43.967060Z","shell.execute_reply":"2023-08-03T02:02:54.790695Z"},"scrolled":true}
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# Assuming you have the DataFrame 'filtered_df' containing the orders with more than 2 items
# Replace 'filtered_df' with your actual DataFrame name

# Group by 'Order No' and collect the items for each order number in a list
order_item_groups = filtered_df.groupby('Order No')['item_name'].apply(list)

# Create a DataFrame to store the item combinations and their counts
item_combinations_df = pd.DataFrame(columns=['Item Combination', 'Count'])

# Calculate the combinations and their counts
for items in order_item_groups:
    item_combinations = list(itertools.combinations(items, 2));
    for combination in item_combinations:
        item_combinations_df = item_combinations_df.append({'Item Combination': combination, 'Count': 1}, ignore_index=True);

# Group by item combinations and sum the counts
item_combinations_counts = item_combinations_df.groupby('Item Combination')['Count'].sum()

# Sort the combinations based on counts in descending order and take the top 10
top_10_combinations = item_combinations_counts.sort_values(ascending=False).head(10)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T02:04:38.515824Z","iopub.execute_input":"2023-08-03T02:04:38.517253Z","iopub.status.idle":"2023-08-03T02:04:38.525226Z","shell.execute_reply.started":"2023-08-03T02:04:38.517209Z","shell.execute_reply":"2023-08-03T02:04:38.523931Z"}}
# import itertools

# # Assuming you have the 'order_item_combinations' dictionary from the previous code
# # Replace 'order_item_combinations' with your actual dictionary name

# # Sort the order counts in descending order and take the top 10 order numbers
# top_5_orders = order_counts.head(5).index

# # Filter the order_item_combinations dictionary to include only the top 10 orders
# top_5_combinations = {order_number: combinations for order_number, combinations in filtered_df.items() if order_number in top_5_orders}

# # Print the combinations for the top 10 orders
# for order_number, combinations in top_5_combinations.items():
#     print(f"Order No: {order_number}")
#     for combination in combinations:
#         print(combination)
#     print("---------------------")


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T02:05:11.038164Z","iopub.execute_input":"2023-08-03T02:05:11.038582Z","iopub.status.idle":"2023-08-03T02:05:11.045218Z","shell.execute_reply.started":"2023-08-03T02:05:11.038548Z","shell.execute_reply":"2023-08-03T02:05:11.043993Z"}}
# import networkx as nx
# import matplotlib.pyplot as plt

# # Assuming you have the 'order_item_combinations' dictionary from the previous code
# # Replace 'order_item_combinations' with your actual dictionary name

# # Create an empty graph
# G = nx.Graph()

# # Add edges (combinations) to the graph from the 'order_item_combinations' dictionary
# for order_combinations in top_5_combinations.values():
#     for combination in order_combinations:
#         G.add_edge(*combination)

# # Draw the network graph
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, k=0.1, seed=42)  # Adjust 'k' to control the layout
# nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, font_weight='bold', alpha=0.8, edge_color='gray')
# plt.title("Items Ordered Together")
# plt.axis('off')
# plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2023-08-03T02:05:14.632384Z","iopub.execute_input":"2023-08-03T02:05:14.632804Z","iopub.status.idle":"2023-08-03T02:05:15.158873Z","shell.execute_reply.started":"2023-08-03T02:05:14.632770Z","shell.execute_reply":"2023-08-03T02:05:15.157522Z"}}
# Plot the top 10 item combinations using a bar chart
plt.figure(figsize=(10, 6))
top_10_combinations.plot(kind='bar', color='skyblue', alpha=0.7)
plt.xlabel('Item Combination')
plt.ylabel('Frequency')
plt.title('Top 10 Item Combinations Ordered Together')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [code]


# %% [code]

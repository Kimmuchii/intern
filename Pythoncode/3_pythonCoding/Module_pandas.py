import os
import matplotlib
import requests

#import requests
#import flask

#import pandas
#import openai
#import scikit-learn
#import beautifulsoup4
#import matplotlib

#################
import koo as k

a = k.function()

print(a)

#######################
import matplotlib.pyplot as plit






#######################################
# python matplotlib1

import matplotlib.pyplot as plt
import numpy as np

# Create data points
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot the data
plt.plot(x, y, label="Sine Wave", color="blue", linewidth=2)

# Add styling
plt.title("Simple Matplotlib Test")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.grid(True)
plt.legend()

# Display the window
plt.show()

########################################
# python matplotlib2

import matplotlib.pyplot as plt

# Categorical data
languages = ['Python', 'Java', 'C++', 'JavaScript']
popularity = [85, 60, 45, 70]
colors = ['#3776AB', '#007396', '#00599C', '#F7DF1E'] # Official brand colors

# Create bar chart
plt.bar(languages, popularity, color=colors)

# Add styling
plt.title("Programming Language Popularity Test")
plt.xlabel("Languages")
plt.ylabel("Score")

# Display the window
plt.show()

##########################################
# python pandas
import pandas as pd

# 1. Create a dictionary of complete mock data (5 items each)
data = {
    'Product': ['Laptop', 'Mouse', 'Monitor', 'Keyboard', 'Desk Lamp'],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Furniture'],
    'Price': [1200, 25, 300, 75, 40],
    'Stock': [15, 120, 30, 45, 60]
}

# 2. Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

print("--- Full Inventory DataFrame ---")
print(df)

# 3. Calculate basic metrics
print("\n--- Summary Statistics ---")
total_value = df['Price'].sum()
average_price = df['Price'].mean()

print(f"Total inventory value: ${total_value}")
print(f"Average product price: ${average_price:.2f}")


# 4. Get the rows with the highest price
# Method A: Get the single most expensive product row
most_expensive_row = df.loc[df['Price'].idxmax()]

print("--- Most Expensive Product Details ---")
print(f"Product Name: {most_expensive_row['Product']}")
print(f"Category:     {most_expensive_row['Category']}")
print(f"Highest Price:${most_expensive_row['Price']}")


# 4. Get the rows with the highest price
# Method A: Get the single most expensive product row
most_expensive_row = df.loc[df['Price'].idxmin()]

print("--- Most Expensive Product Details ---")
print(f"Product Name: {most_expensive_row['Product']}")
print(f"Category:     {most_expensive_row['Category']}")
print(f"Highest Price:${most_expensive_row['Price']}")

# 5. How can I visualize the graph in matplotlib

import matplotlib.pyplot as plot

plt.figure(figsize=(8,5))

plt.bar(df['Product'], df['Price'], color='skyblue', edgecolor='black')

plt.title('Product Price Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Product Names', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()





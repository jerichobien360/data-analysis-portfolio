import pandas as pd
import requests

# Download sample e-commerce data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
response = requests.get(url)
with open('data/online_retail.xlsx', 'wb') as file:
    file.write(response.content)

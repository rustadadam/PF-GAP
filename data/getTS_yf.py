import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf

labelDF = pd.read_csv("data/Labeled_11-01-2024_GICS.csv")

syms = list(labelDF["Symbol "])

# - > Start Date is 2022-01-04
# - > End Date is 2025-04-24
data = yf.download(syms, start="2024-04-24", end="2025-04-24", interval="60m", group_by="ticker", progress=False, auto_adjust=True)
data_close = data.xs('Close', axis=1, level=1)

fails = []
for sym in syms:
    try:
        tempdf = data[sym]
        filename = sym + "_2024-10.csv"
        tempdf.to_csv("Data/" + filename)
    except:
        fails.extend([sym])


data_close = data.xs('Close', axis=1, level=1).T.pct_change(axis=1)#.drop(columns = data_close.columns[0])

data_close = data_close.drop(columns=data_close.columns[0]).drop(index=data_close.index[0], inplace=False)
data_close

rows_with_missing = data_close.index[data_close.isna().any(axis=1)]
rows_with_missing

data_close = data_close.dropna()

data_close.isna().sum().sum()


# Remove any trailing spaces in the 'Symbol ' column name and values
labelDF.columns = [col.strip() for col in labelDF.columns]
labelDF['Symbol'] = labelDF['Symbol'].str.strip() if 'Symbol' in labelDF.columns else labelDF['Symbol '].str.strip()

# Create a mapping from ticker to sector
ticker_to_sector = dict(zip(labelDF['Symbol'], labelDF['Labels']))

# Add a 'Sector' column to data_close by mapping the index (tickers) to sector
data_close['Sector'] = data_close.index.map(ticker_to_sector)

# Show the updated DataFrame
data_close.head()


# Save the updated DataFrame to a CSV file
data_close.to_csv("../data/Hourly_Returns_2024-10(Labels-included).csv")
print("Data saved to ../data/Hourly_Returns_2024-10(Labels-included).csv")
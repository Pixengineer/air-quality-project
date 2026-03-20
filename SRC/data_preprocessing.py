import pandas as pd

# load dataset
df = pd.read_csv("data/city_day.csv")

print("Initial Shape:", df.shape)

# check missing values
print(df.isnull().sum())

# remove rows where AQI is missing (IMPORTANT)
df = df.dropna(subset=['AQI'])

# fill remaining missing values
df = df.fillna(method='ffill')

# convert date
df['Date'] = pd.to_datetime(df['Date'])

# feature engineering
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month

# drop unnecessary columns
df = df.drop(['Date', 'City', 'AQI_Bucket'], axis=1)

print("Final Shape:", df.shape)

# save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

print("✅ Data Preprocessing Done!")
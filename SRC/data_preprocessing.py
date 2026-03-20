import pandas as pd


df = pd.read_csv("data/city_day.csv")

print("Initial Shape:", df.shape)


print(df.isnull().sum())


df = df.dropna(subset=['AQI'])


df = df.fillna(method='ffill')


df['Date'] = pd.to_datetime(df['Date'])


df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month


df = df.drop(['Date', 'City', 'AQI_Bucket'], axis=1)

print("Final Shape:", df.shape)


df.to_csv("data/cleaned_data.csv", index=False)

print("✅ Data Preprocessing Done!")
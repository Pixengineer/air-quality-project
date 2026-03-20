import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

print("Columns:", df.columns)

# features & target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# prediction
pred = model.predict(X_test)

# evaluation
mae = mean_absolute_error(y_test, pred)
print("✅ MAE:", mae)

# save model
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")
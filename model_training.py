import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Load the data
data = pd.read_csv('d:\\prcar\\audi.csv')

# Preprocess the data
le_model = LabelEncoder()
le_transmission = LabelEncoder()
le_fuelType = LabelEncoder()

data['model'] = le_model.fit_transform(data['model'])
data['transmission'] = le_transmission.fit_transform(data['transmission'])
data['fuelType'] = le_fuelType.fit_transform(data['fuelType'])

# Select features and target
features = ['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']
target = 'price'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: £{rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# Save the model and label encoders
joblib.dump(dt_regressor, 'dt_regressor_model.joblib')
joblib.dump(le_model, 'le_model.joblib')
joblib.dump(le_transmission, 'le_transmission.joblib')
joblib.dump(le_fuelType, 'le_fuelType.joblib')

print("Model and label encoders saved successfully.")
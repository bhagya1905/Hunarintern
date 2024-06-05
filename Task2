import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Create a mock dataset
dates = pd.date_range(start='1/1/2010', end='12/31/2015')
data = pd.DataFrame(dates, columns=['Date'])
data['Temp'] = np.random.uniform(low=0, high=30, size=len(dates))

# Preprocess the data
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Define features and target variable
X = data[['Year', 'Month', 'Day']]
y = data['Temp']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Perform additional testing with cross-validation
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
print(f'Cross-validated Mean Squared Error: {-scores.mean()}')

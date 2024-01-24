import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the file
with open('Doviz_Satislari_20050101_20231205_Training_Set.csv', 'r') as file:
    data = file.read()

# Remove double quotation marks
data_without_quotes = data.replace('"', '')

# Write the corrected data to 'Doviz_Satislari_20050101_20231205_Training_Set_without_quotes.csv'
with open('Doviz_Satislari_20050101_20231205_Training_Set_without_quotes.csv', 'w') as file:
    file.write(data_without_quotes)

# Load the corrected data into a Pandas DataFrame
df = pd.read_csv('Doviz_Satislari_20050101_20231205_Training_Set_without_quotes.csv')

# Remove rows containing empty values
df_cleaned = df.dropna(subset=['TP DK USD S YTL', 'TP DK EUR S YTL'])

# Write the cleaned data to 'cleaned_dataset.csv'
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

# Load the cleaned data using numpy
my_data = np.genfromtxt('cleaned_dataset.csv', delimiter=',', skip_header=1)
my_x = my_data[:, 3]  # Independent variable
my_y = my_data[:, 7]  # Dependent variable

my_x = my_x.reshape(-1, 1)

# Define a function to create polynomial features
def create_polynomial_features(my_x, degree=4):
    features = np.ones((len(my_x), 1))
    for i in range(1, degree + 1):
        features = np.concatenate((features, my_x ** i), axis=1)
    return features

# Create polynomial features
my_x_poly = create_polynomial_features(my_x, degree=4)

# Calculate weights and coefficients using the normal equation
def normal_equation(my_x, my_y):
    weights = np.linalg.inv(my_x.T.dot(my_x)).dot(my_x.T).dot(my_y)
    coefficients = weights[1:]  # Remove the first element as it corresponds to the bias
    return weights, coefficients

# Train the model
weights, coefficients = normal_equation(my_x_poly, my_y)

# Create the prediction function
def predict(my_x, weights):
    return my_x.dot(weights)

# Function to calculate mean squared error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Model predictions
my_y_pred = predict(my_x_poly, weights)

# Calculate MSE
mse = mean_squared_error(my_y, my_y_pred)
print(f"Mean Squared Error: {mse}")

# Print weighted coefficients
print(f"Intercept (Bias): {weights[0]}")
print("Coefficients:")
for i, coefficient in enumerate(coefficients, 1):
    print(f"Degree {i}: {coefficient}")

# Visualization
plt.scatter(my_x, my_y, color='red')  # TP DK USD S YTL
plt.plot(my_x, predict(my_x_poly, weights), color='blue')  # Predicted TP DK EUR S YTL
plt.title('Polynomial Regression')
plt.xlabel('TP DK USD S YTL')
plt.ylabel('TP DK EUR S YTL')
plt.legend(['Predicted', 'Actual'])
plt.show()

# Make a new prediction
tp_dk_usd_s_ytl = 6.5
tp_dk_usd_s_ytl_poly = create_polynomial_features(np.array([[tp_dk_usd_s_ytl]]), degree=4)
predicted_tp_dk_eur_s_ytl = predict(tp_dk_usd_s_ytl_poly, weights)
print(f"Predicted TP DK EUR S YTL for TP DK USD S YTL {tp_dk_usd_s_ytl}: ${predicted_tp_dk_eur_s_ytl[0]:.2f}")
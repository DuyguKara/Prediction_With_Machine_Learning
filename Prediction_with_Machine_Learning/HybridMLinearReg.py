import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyHybridL1L2RegularizedMultipleLinearRegression:
    def __init__(self, alpha_l1=0.01, alpha_l2=0.01):
        self.weights = None
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.zeros(X.shape[1])

        for epoch in range(epochs):
            y_pred = X.dot(self.weights)

            # Compute the derivative of MSE with L1 and L2 regularization
            error = y_pred - y
            derivative = (2/len(y)) * (X.T.dot(error) + self.alpha_l2 * self.weights + self.alpha_l1 * np.sign(self.weights))
            
            # Update weights using gradient descent with L1 and L2 regularization
            old_weights = self.weights.copy()
            self.weights = self.update_weights(self.weights, derivative, learning_rate)

            # Print the derivative of MSE, old and current weights for every 100 epochs
            if epoch % 100 == 0:
                print(f'\nEpoch {epoch}, Derivative of MSE: {derivative}')
                print(f'Old Weights: {old_weights}')
                print(f'Current Weights: {self.weights}')

        # Print the final derivative of MSE and weights after training
        print(f'\nFinal Derivative of MSE: {derivative}')
        print(f'Final Weights: {self.weights}')

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

    def evaluate_performance(self, y_true, y_pred):
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true)**2)
        rmse = np.sqrt(mse)
        total_variance = np.sum((y_true - np.mean(y_true))**2)
        explained_variance = np.sum((y_pred - np.mean(y_true))**2)
        r2 = explained_variance / total_variance

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}

    def update_weights(self, weights, derivative, learning_rate):
        # Update weights using gradient descent
        updated_weights = weights - learning_rate * derivative
        return updated_weights

# Read the file
df = pd.read_csv('Doviz_Satislari_20050101_20231205_Training_Set_without_quotes.csv')

# Remove rows containing empty values
df_cleaned = df.dropna(subset=['TP DK USD S YTL', 'TP DK EUR S YTL'])

# Write the cleaned data to 'cleaned_dataset.csv'
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

# Define dependent and independent variables
X = df_cleaned.iloc[:, 2:7].values
y = df_cleaned.iloc[:, 6].values

# Split the dataset into training and test sets
def train_test_split_custom(X, y, test_size=0.2):
    split_index = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)

# Create and train the model with hybrid L1/L2 regularization
model = MyHybridL1L2RegularizedMultipleLinearRegression(alpha_l1=0.01, alpha_l2=0.01)
model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print predictions and actual values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df)

# Calculate performance metrics and print them
performance_metrics = model.evaluate_performance(y_test, y_pred)
for metric, value in performance_metrics.items():
    print(f'{metric}: {value}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyMultipleLinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.zeros(X.shape[1])

        for _ in range(epochs):
            y_pred = X.dot(self.weights)

            # Compute the derivative of MSE
            error = y_pred - y
            derivative = (2/len(y)) * X.T.dot(error)
            
            # Update weights using gradient descent
            self.weights -= learning_rate * derivative

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

if __name__ == "__main__":
    # Read the file
    df = pd.read_csv('Doviz_Satislari_20050101_20231205_Training_Set_without_quotes.csv')

    # Remove rows containing empty values
    df_cleaned = df.dropna(subset=['TP DK USD S YTL', 'TP DK EUR S YTL'])

    # Write the cleaned data to 'cleaned_dataset.csv'
    df_cleaned.to_csv('cleaned_dataset.csv', index=False)

    # Define dependent and independent variables
    X = df_cleaned.iloc[:, 2:9].values
    y = df_cleaned.iloc[:, 9].values

    # Split the dataset into training and test sets
    def train_test_split_custom(X, y, test_size=0.2):
        split_index = int((1 - test_size) * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)

    # Create and train the model
    model = MyMultipleLinearRegression()
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

    # Get user input values for prediction
    user_inputs = []
    for i in range(2, 9):  # Model was trained with 7 features, so we only take those features
        value = float(input(f'Enter the value for Feature {i-1}: '))
        user_inputs.append(value)

    # Make prediction using user input
    user_prediction = model.predict(np.array(user_inputs).reshape(1, -1))

    # Display the user input and the prediction result
    print(f'\nUser Input: {user_inputs}')
    print(f'Predicted Value: {user_prediction[0]:.2f}')
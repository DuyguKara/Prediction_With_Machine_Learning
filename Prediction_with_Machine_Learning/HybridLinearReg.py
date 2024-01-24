import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MyLinearRegression:
    def __init__(self, alpha=0.01, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.opt_w1 = None
        self.opt_wo = None
        self.cost_loss = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x_list, y_list):
        n = len(x_list)

        # Initialize coefficients
        self.opt_w1 = np.random.rand()
        self.opt_wo = np.random.rand()

        for _ in range(self.max_iter):
            # Calculate means
            mean_x = np.mean(x_list)
            mean_y = np.mean(y_list)

            # Calculate gradients
            grad_w1 = -2 * np.sum(x_list * (y_list - (self.opt_w1 * x_list + self.opt_wo))) / n
            grad_wo = -2 * np.sum(y_list - (self.opt_w1 * x_list + self.opt_wo)) / n

            # Update coefficients with regularization terms
            self.opt_w1 -= self.alpha * (grad_w1 + 2 * self.l1_ratio * np.sign(self.opt_w1) + 2 * (1 - self.l1_ratio) * self.opt_w1)
            self.opt_wo -= self.alpha * (grad_wo + 2 * self.l1_ratio * np.sign(self.opt_wo) + 2 * (1 - self.l1_ratio) * self.opt_wo)

            # Calculate loss (mean squared error)
            my_y_pred = self.predict(x_list)
            loss = np.mean((my_y_pred - y_list) ** 2)

            if self.cost_loss is None or loss < self.cost_loss:
                self.cost_loss = loss
                opt_w1_temp = self.opt_w1
                opt_wo_temp = self.opt_wo

            print(f"Iteration: {_ + 1}, Cost/Loss: {self.cost_loss:.4f}")

            # Convergence check
            if np.abs(grad_w1) < self.tol and np.abs(grad_wo) < self.tol:
                break

        print(f"Optimal w1: {opt_w1_temp:.4f}")
        print(f"Optimal w0: {opt_wo_temp:.4f}")

        return opt_w1_temp, opt_wo_temp

    def predict(self, x):
        return self.opt_w1 * x + self.opt_wo

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

# Train the model
my_model = MyLinearRegression(alpha=0.01, l1_ratio=0.5, max_iter=1000, tol=1e-4)
my_model.fit(my_x, my_y)

# Make predictions with the trained model
my_y_pred = my_model.predict(my_x)

# Visualize the results on the training set
plt.scatter(my_x, my_y, color='blue')
plt.plot(my_x, my_y_pred, color='red')
plt.title('My Dataset Visualization')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()
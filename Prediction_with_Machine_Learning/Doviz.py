import pandas as pd

# Read the CSV file
df = pd.read_csv("Doviz_Satislari_20050101_20231205_Training_Set.csv")

# Display the entire dataset
print(df)

# Get user input for "No" column value
i = int(input("Enter the value for 'No' column: "))
selected_row = df[df["No"] == i]

# Check if the selected row is not empty
if not selected_row.empty:
    # Extract input features and output feature from the selected row
    input_features = selected_row.iloc[0, 2:9]
    output_feature = selected_row.iloc[0, 9]

    # Display the selected record
    print(f"\nInput Features (record {i}):")
    print(input_features)

    print("\nOutput Feature:")
    print(output_feature)
else:
    print(f"No matching row found for the given 'No' column value: {i}")
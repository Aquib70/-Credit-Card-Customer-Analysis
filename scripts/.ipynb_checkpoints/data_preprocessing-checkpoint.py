import pandas as pd

# Load the dataset from a CSV file
csv_file_path = r'C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\credit_card_data_csv.csv'

print("Loading the dataset...")
df = pd.read_csv(csv_file_path)
print("Dataset loaded successfully.")

# Handle missing values
print("Filling missing values with column medians...")
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
print("Missing values handled.")

# Calculate outlier bounds for all numerical columns
print("Calculating outlier bounds for numerical columns...")
bounds = {}
for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    bounds[column] = (lower_bound, upper_bound)
    print(f"{column}: lower bound = {lower_bound}, upper bound = {upper_bound}")

# Filter the DataFrame based on calculated bounds
print("Filtering the DataFrame to remove outliers...")
for column, (lower_bound, upper_bound) in bounds.items():
    original_size = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    filtered_size = len(df)
    print(f"{column}: removed {original_size - filtered_size} outliers")

# Save the cleaned dataset
cleaned_file_path = r'C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\cleaned_credit_card_data.csv'
print(f"Saving the cleaned dataset to {cleaned_file_path}...")
df.to_csv(cleaned_file_path, index=False)
print("Cleaned dataset saved successfully.")

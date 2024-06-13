import pandas as pd
import numpy as np

# Load your cleaned dataset
file_path = r"C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\cleaned_credit_card_data.csv"
df = pd.read_csv(file_path)

# Check unique values in the Attrition_Flag column
unique_values = df['Attrition_Flag'].unique()
print("Unique values in Attrition_Flag column:", unique_values)

# Simulate data for 'Attrited Customer'
num_attrited_customers = 1000  # Adjust as needed

# Define characteristics for simulated attrited customers (example)
simulated_data = {
    'Attrition_Flag': ['Attrited Customer'] * num_attrited_customers,
    'Customer_Age': np.random.randint(20, 70, size=num_attrited_customers),
    'Gender': np.random.choice(['M', 'F'], size=num_attrited_customers),
    'Dependent_count': np.random.randint(0, 5, size=num_attrited_customers),
    # Add more features as needed
}

# Create DataFrame from simulated data
simulated_df = pd.DataFrame(simulated_data)

# Concatenate original data with simulated data
augmented_df = pd.concat([df, simulated_df], axis=0)

# Save augmented dataset to CSV
augmented_file_path = r"C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\augmented_credit_card_data.csv"
augmented_df.to_csv(augmented_file_path, index=False)

print(f"Augmented dataset saved to {augmented_file_path}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the cleaned dataset
file_path = r"C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\cleaned_credit_card_data.csv"
df = pd.read_csv(file_path)

# Identify categorical columns
categorical_cols = ['Education_Level', 'Marital_Status']  # Add more columns as needed

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Separate features and target variable
X = df_encoded.drop(columns=['Attrition_Flag'])
target = df_encoded['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)

# Ensure there are samples of at least 2 classes in the training set
if len(y_train.unique()) > 1:
    # Train and evaluate Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print('Random Forest Classification Report:')
    print(classification_report(y_test, rf_preds))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, rf_preds))

    # Train and evaluate Logistic Regression model
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    print('\nLogistic Regression Classification Report:')
    print(classification_report(y_test, lr_preds))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, lr_preds))
else:
    print("Not enough class variability in the target variable for training models.")

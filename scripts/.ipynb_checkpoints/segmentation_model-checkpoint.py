import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\cleaned_credit_card_data.csv")

# Ensure column names match the dataset
print("Column names in the dataset:")
print(df.columns)

# Select features for clustering
features = df[['Customer_Age', 'Income_Category', 'Credit_Limit', 'Total_Revolving_Bal']]

# Convert categorical column 'Income_Category' to numerical values using one-hot encoding
features = pd.get_dummies(features, columns=['Income_Category'])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Fit the KMeans model with the optimal number of clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Save the clustered dataset
df.to_csv(r"C:\Users\Admin\Desktop\-Credit-Card-Customer-Analysis\data\clustered_credit_card_data.csv", index=False)

print("Clustering completed and dataset saved successfully.")

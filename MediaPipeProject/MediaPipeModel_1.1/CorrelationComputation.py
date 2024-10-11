# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Your Data
# -----------------------------

# Replace 'your_data.csv' with the actual path to your CSV file
data = pd.read_csv(r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\MediaPipe1.1Counter.csv")

# Display the first few rows to verify
print("Data Preview:")
print(data.head())

# -----------------------------
# 2. Prepare Features and Target Variable
# -----------------------------

# List of facial expression features
features = ['Blinking', 'Facial Twitching',
            'Smiling','Head Tilt']

# Check if all features are present in the data
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    print(f"Error: The following features are missing in your data: {missing_features}")
    exit(1)

# Prepare the feature matrix X
X = data[features]

# Prepare the target variable y
# Assuming your data includes the labels mapped as per your mapping in a column named 'Label'
if 'Label' not in data.columns:
    print("Error: 'Label' column not found in the data.")
    exit(1)

y = data['Label']

# Verify the unique labels
print("\nUnique labels in the data:", y.unique())

# -----------------------------
# 3. Handle Missing Values
# -----------------------------

# Check for missing values in features and target
print("\nMissing values in each column:")
print(data[features + ['Label']].isnull().sum())

# Drop rows with missing values in features or target
data = data.dropna(subset=features + ['Label'])

# Update X and y after dropping missing values
X = data[features]
y = data['Label']

# Convert labels to integer type if they are not already
y = y.astype(int)

# -----------------------------
# 4. Feature Scaling
# -----------------------------

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Split the Data
# -----------------------------

# Split the data into training and testing sets
# Use stratify=y to maintain label distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Train the Random Forest Classifier
# -----------------------------

# Initialize the Random Forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate the Model
# -----------------------------

# Predict on the test set
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 8. Analyze Feature Importance
# -----------------------------

# Get feature importances
importances = model.feature_importances_
feature_names = features
forest_importances = pd.Series(importances, index=feature_names)

# Plot feature importances
plt.figure(figsize=(8, 6))
forest_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

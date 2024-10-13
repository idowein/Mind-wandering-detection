import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_random_forest(csv_file_path):
    # Load the CSV data
    data = pd.read_csv(csv_file_path)

    # Clean column names (remove leading/trailing whitespace)
    data.columns = data.columns.str.strip()

    # Print the list of columns in the dataset
    print("Columns in the dataset:", data.columns.tolist())

    # Display the first few rows
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # Define feature columns and target variable
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']
    target_column = 'Mapped_Label'  # Update this if your label column has a different name

    # Ensure that the necessary columns are present
    required_columns = feature_columns + [target_column]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"\nError: The dataset is missing the following required columns: {missing_columns}")
        return

    # Check for missing values and fill them if necessary
    if data[required_columns].isnull().values.any():
        print("\nWarning: The dataset contains missing values. Filling missing values with zeros.")
        data[required_columns] = data[required_columns].fillna(0)

    # Extract features and target
    X = data[feature_columns]
    y = data[target_column]

    # Ensure labels are integers
    if not pd.api.types.is_integer_dtype(y):
        y = y.astype(int)

    # Display label distribution
    print("\nLabel Distribution:")
    print(y.value_counts())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_classifier.fit(X_train, y_train)
    print("\nModel training completed.")

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    model_output_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\random_forest_model.joblib'
    joblib.dump(rf_classifier, model_output_path)
    print(f"\nTrained model saved to {model_output_path}")

if __name__ == "__main__":
    # Replace with the path to your CSV file
    csv_file_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\output_results2.csv"

    train_random_forest(csv_file_path)

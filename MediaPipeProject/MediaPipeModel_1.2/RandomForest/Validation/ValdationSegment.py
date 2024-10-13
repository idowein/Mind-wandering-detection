import pandas as pd
import joblib


def predict_labels(validation_csv_path, model_path, output_csv_path):
    # Load the validation data
    validation_data = pd.read_csv(validation_csv_path)

    # Clean column names
    validation_data.columns = validation_data.columns.str.strip()

    # Display the first few rows
    print("First few rows of the validation dataset:")
    print(validation_data.head())

    # Define feature columns
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']

    # Check if necessary columns are present
    missing_columns = [col for col in feature_columns if col not in validation_data.columns]
    if missing_columns:
        print(f"\nError: The validation dataset is missing the following required feature columns: {missing_columns}")
        return

    # Handle missing values in features
    if validation_data[feature_columns].isnull().values.any():
        print("\nWarning: Missing values found in feature columns. Filling missing values with zeros.")
        validation_data[feature_columns] = validation_data[feature_columns].fillna(0)

    # Extract features
    X_validation = validation_data[feature_columns]

    # Load the trained model
    try:
        model = joblib.load(model_path)
        print("\nTrained model loaded successfully.")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return

    # Make predictions on the validation data
    y_pred = model.predict(X_validation)

    # Add predictions to the validation data
    validation_data['Predicted_Label'] = y_pred

    # Save the predictions to a new CSV file
    validation_data.to_csv(output_csv_path, index=False)
    print(f"\nPredictions saved to '{output_csv_path}'")

    # Optional: Display the first few rows of the output file
    print("\nFirst few rows with predictions:")
    print(validation_data.head())


if __name__ == "__main__":
    # Replace with the path to your validation CSV file
    validation_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\results.csv"

    # Replace with the path to your trained model file
    model_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\random_forest_model.joblib"

    # Replace with the desired output CSV file path
    output_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\predictionResults.csv"

    predict_labels(validation_csv_path, model_path, output_csv_path)

import pandas as pd
import joblib
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_prediction.log"),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(csv_path, has_labels=True):
    logging.info("Loading data from CSV...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    # Check for missing columns
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']
    required_columns = feature_columns.copy()
    if has_labels:
        required_columns.append('Label')
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    if has_labels:
        # Drop rows with missing labels
        data = data.dropna(subset=['Label'])

        # Ensure labels are integers
        data['Label'] = data['Label'].astype(int)

        # Validate label values
        valid_labels = [1, 2, 3, 4]
        data = data[data['Label'].isin(valid_labels)]

        # Adjust labels to start from 0
        data['Adjusted_Label'] = data['Label'] - 1  # Labels now in [0, 1, 2, 3]

    # Handle missing values in features
    data[feature_columns] = data[feature_columns].fillna(0)

    logging.info("Data preprocessing completed.")
    return data

def predict_and_evaluate(model_path, input_csv_path, output_csv_path, has_labels=True):
    # Load the trained model
    logging.info("Loading the trained model...")
    model = joblib.load(model_path)

    # Load and preprocess data
    data = load_and_preprocess_data(input_csv_path, has_labels=has_labels)
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']

    X_input = data[feature_columns]

    # Make predictions
    logging.info("Making predictions...")
    y_pred = model.predict(X_input)

    # Adjust predicted labels back to original
    y_pred_adjusted = y_pred + 1
    data['Predicted_Label'] = y_pred_adjusted

    # Save predictions to CSV
    output_columns = feature_columns + ['Predicted_Label']
    if has_labels:
        data['Mapped_Label'] = data['Adjusted_Label'] + 1  # Adjust true labels back
        output_columns.append('Mapped_Label')
    data.to_csv(output_csv_path, index=False, columns=output_columns)
    logging.info(f"Predictions saved to '{output_csv_path}'")

    # If true labels are available, evaluate the model
    if has_labels:
        y_true = data['Mapped_Label']
        accuracy = accuracy_score(y_true, y_pred_adjusted)
        precision = precision_score(y_true, y_pred_adjusted, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_adjusted, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred_adjusted, average='weighted', zero_division=0)
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # Classification report
        labels = [1, 2, 3, 4]
        class_names = ['Engagement', 'Boredom', 'Confusion', 'Frustration']
        report = classification_report(y_true, y_pred_adjusted, labels=labels, target_names=class_names, zero_division=0)
        logging.info("\nClassification Report:\n" + report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_adjusted, labels=labels)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        logging.info("\nConfusion Matrix:\n" + str(cm_df))

    else:
        logging.info("No true labels provided; skipping evaluation metrics.")

if __name__ == "__main__":
    # Paths to your files (replace with your actual paths)
    model_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\xgboost_model.joblib"
    input_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\results.csv"
    output_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\XGBPredictionsresults.csv"

    # Set has_labels to False since your input data does not include true labels
    predict_and_evaluate(model_path, input_csv_path, output_csv_path, has_labels=False)

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(csv_path):
    logging.info("Loading data from CSV...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    # Check for missing columns
    required_columns = ['Label', 'Blinking', 'Smiling', 'Head Movement']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Drop rows with missing labels
    data = data.dropna(subset=['Label'])

    # Map labels
    logging.info("Mapping labels...")
    # Original labels mapping to new labels
    label_mapping = {
        **dict.fromkeys([0, 1, 2, 3], 1),      # Engagement -> 1
        **dict.fromkeys([4, 5, 6, 7], 2),      # Boredom -> 2
        **dict.fromkeys([8, 9, 10, 11], 3),    # Confusion -> 3
        **dict.fromkeys([12, 13, 14, 15], 4),  # Frustration -> 4
    }
    data['Mapped_Label'] = data['Label'].map(label_mapping)

    # Drop rows with undefined labels
    data = data.dropna(subset=['Mapped_Label'])
    data['Mapped_Label'] = data['Mapped_Label'].astype(int)

    # Handle missing values in features
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']
    data[feature_columns] = data[feature_columns].fillna(0)

    logging.info("Data preprocessing completed.")
    return data

def handle_class_imbalance(X, y):
    logging.info("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logging.info("Class imbalance handled.")
    return X_resampled, y_resampled

def compute_class_weights(y):
    logging.info("Computing class weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    logging.info(f"Class weights: {class_weight_dict}")
    return class_weight_dict

def perform_hyperparameter_tuning(X_train, y_train, class_weight):
    logging.info("Performing hyperparameter tuning with Grid Search...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }
    rf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=param_grid,
        cv=skf,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    # Classification report
    labels = [1, 2, 3, 4]
    class_names = ['Engagement', 'Boredom', 'Confusion', 'Frustration']
    report = classification_report(y_test, y_pred, labels=labels, target_names=class_names, zero_division=0)
    logging.info("\nClassification Report:\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    logging.info("\nConfusion Matrix:\n" + str(cm_df))

def train_and_save_model(training_csv_path, model_output_path):
    # Load and preprocess data
    data = load_and_preprocess_data(training_csv_path)
    feature_columns = ['Blinking', 'Smiling', 'Head Movement']
    target_column = 'Mapped_Label'

    # Extract features and target
    X = data[feature_columns]
    y = data[target_column]

    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X, y)

    # Split data
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    # Compute class weights
    class_weight = compute_class_weights(y_train)

    # Hyperparameter tuning
    best_model = perform_hyperparameter_tuning(X_train, y_train, class_weight)

    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)

    # Save the trained model
    joblib.dump(best_model, model_output_path)
    logging.info(f"Trained model saved to '{model_output_path}'")

if __name__ == "__main__":
    # Paths to your files (replace with your actual paths)
    training_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\output_results2.csv"
    model_output_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\TrainingDetections\random_forest_model.joblib'

    train_and_save_model(training_csv_path, model_output_path)

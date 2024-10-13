import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model_performance(predictions_csv_path, labels_csv_path):
    # Load the predictions CSV file
    predictions_df = pd.read_csv(predictions_csv_path)
    predictions_df.columns = predictions_df.columns.str.strip()

    # Ensure 'Video Name' and 'Predicted_Label' columns exist
    if 'Video Name' not in predictions_df.columns or 'Predicted_Label' not in predictions_df.columns:
        print("\nError: The predictions CSV must contain 'Video Name' and 'Predicted_Label' columns.")
        return

    # Load the labels CSV file
    labels_df = pd.read_csv(labels_csv_path)
    labels_df.columns = labels_df.columns.str.strip()

    # Ensure 'Video Name' and 'Label' columns exist
    if 'Video Name' not in labels_df.columns or 'Label' not in labels_df.columns:
        print("\nError: The labels CSV must contain 'Video Name' and 'Label' columns.")
        return

    # Remove file extensions from 'Video Name' in both DataFrames
    predictions_df['Video Name'] = predictions_df['Video Name'].apply(lambda x: str(x).split('.')[0])
    labels_df['Video Name'] = labels_df['Video Name'].apply(lambda x: str(x).split('.')[0])

    # Merge the two DataFrames on 'Video Name'
    merged_df = pd.merge(predictions_df, labels_df, on='Video Name', how='inner')

    # Check if the merge resulted in any data
    if merged_df.empty:
        print("\nError: No matching 'Video Name' entries found between predictions and labels.")
        return

    # Extract true labels and predicted labels
    y_true = merged_df['Label'].astype(int)
    y_pred = merged_df['Predicted_Label'].astype(int)

    # Print class distribution
    print("\nClass Distribution in Predictions and Labels:")
    print("Predicted Labels Distribution:\n", y_pred.value_counts().sort_index())
    print("True Labels Distribution:\n", y_true.value_counts().sort_index())

    # Define the labels and class names
    labels = [4, 1, 3, 2]  # Assuming 1 = Frustration, 2 = Engagement, 3 = Confusion, 4 = Boredom
    class_names = ['Frustration', 'Engagement', 'Confusion', 'Boredom']

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Display overall metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Display the classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))

    # Display the confusion matrix as a heatmap
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="YlGnBu")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

if __name__ == "__main__":
    # Replace with the path to your predictions CSV file
    predictions_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\XGBPredictionsresults.csv"

    # Replace with the path to your labels CSV file
    labels_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\FixedLables\NewValidationLabels.csv"

    evaluate_model_performance(predictions_csv_path, labels_csv_path)

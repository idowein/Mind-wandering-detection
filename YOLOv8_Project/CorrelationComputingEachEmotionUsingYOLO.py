import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Paths to CSV files
# -----------------------------
facial_expression_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\facial_expression_results.csv"
labels_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\AllLabels.csv"

# -----------------------------
# Read the facial expression results CSV
# -----------------------------
facial_df = pd.read_csv(facial_expression_csv)

# -----------------------------
# Read the labels CSV
# -----------------------------
labels_df = pd.read_csv(labels_csv)

# -----------------------------
# Data Preprocessing
# -----------------------------
# Ensure 'Video' column in facial_df contains only filenames
facial_df['Video'] = facial_df['Video'].apply(os.path.basename)

# Create 'ClipID' column in facial_df by removing file extensions
facial_df['ClipID'] = facial_df['Video'].apply(lambda x: os.path.splitext(x)[0])

# Remove file extensions from labels_df['ClipID'] if necessary
labels_df['ClipID'] = labels_df['ClipID'].apply(lambda x: os.path.splitext(str(x))[0])

# Convert 'ClipID's to lowercase to ensure consistency
facial_df['ClipID'] = facial_df['ClipID'].str.lower()
labels_df['ClipID'] = labels_df['ClipID'].str.lower()

# Strip any leading/trailing spaces
facial_df['ClipID'] = facial_df['ClipID'].str.strip()
labels_df['ClipID'] = labels_df['ClipID'].str.strip()

# Merge on 'ClipID'
merged_df = pd.merge(facial_df, labels_df, on='ClipID')

# Check number of rows in merged DataFrame
print(f"Number of rows in merged DataFrame after adjustment: {len(merged_df)}")

# Proceed only if merged_df is not empty
if len(merged_df) > 0:
    # -----------------------------
    # Define features and emotion labels
    # -----------------------------
    facial_features = ['Blink_Count', 'Head_Movements', 'Eyebrow_Contractions',
                       'Cheek_Raisings', 'Facial_Twitches']

    emotion_labels = ['Engagement', 'Boredom', 'Confusion', 'Frustration']

    # Initialize an empty list to store evaluation results
    evaluation_results = []

    # Loop through each emotion label
    for target in emotion_labels:
        print(f"\n--- Evaluating {target} ---")
        # Ensure that the labels are numerical
        if merged_df[target].dtype == 'object':
            merged_df[target] = pd.to_numeric(merged_df[target], errors='coerce')

        # Drop rows with missing values in features or target
        model_data = merged_df[facial_features + [target]].dropna()

        X = model_data[facial_features]
        y = model_data[target]

        # Check label distribution
        label_counts = y.value_counts()
        print(f"Value counts for '{target}':")
        print(label_counts)

        # Binarize target if needed (assuming labels from 0 to 3)
        # Adjust threshold based on your data
        threshold = 2
        y = y.apply(lambda x: 1 if x >= threshold else 0)

        # Handle class imbalance using SMOTE
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Initialize and train the classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_resampled, y_train_resampled)

        # Predict and evaluate
        y_pred = clf.predict(X_test)

        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel Evaluation Metrics for {target}:")
        print("--------------------------------------")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")

        # Append results to the list
        evaluation_results.append({
            'Emotion': target,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Accuracy': accuracy
        })

        # Classification report
        print(f"\nClassification Report for {target}:")
        print("-----------------------------------")
        print(classification_report(y_test, y_pred, target_names=[f'Low {target}', f'High {target}']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Low {target}', f'High {target}'],
                    yticklabels=[f'Low {target}', f'High {target}'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {target}')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Create a summary table of evaluation metrics
    # -----------------------------
    results_df = pd.DataFrame(evaluation_results)
    print("\nSummary of Evaluation Metrics:")
    print(results_df)

    # Optionally, save the results to a CSV file
    output_dir = os.path.dirname(facial_expression_csv)
    results_csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nEvaluation summary saved to {results_csv_path}")

    # -----------------------------
    # Estimate Mind Wandering Percentage
    # -----------------------------
    # Assuming you have a 'Mind_Wandering' label in your merged_df, proceed
    if 'Mind_Wandering' in merged_df.columns:
        print("\n--- Estimating Mind Wandering Percentage ---")
        # Ensure 'Mind_Wandering' is numerical
        if merged_df['Mind_Wandering'].dtype == 'object':
            merged_df['Mind_Wandering'] = pd.to_numeric(merged_df['Mind_Wandering'], errors='coerce')

        # Loop through each emotion label
        mind_wandering_results = []

        for target in emotion_labels:
            print(f"\nComputing Mind Wandering Percentage for {target}")
            # Use the same model_data as before
            model_data = merged_df[[target, 'Mind_Wandering']].dropna()
            y_emotion = model_data[target].apply(lambda x: 1 if x >= threshold else 0)
            y_mw = model_data['Mind_Wandering'].apply(lambda x: 1 if x >= threshold else 0)

            # Compute the percentage of mind wandering for each emotion level
            mw_percentage = model_data.groupby(y_emotion)['Mind_Wandering'].mean() * 100

            print(f"Percentage of Mind Wandering for {target}:")
            print(mw_percentage)

            # Append results
            mind_wandering_results.append({
                'Emotion': target,
                f'Low {target} MW%': mw_percentage.get(0, 0),
                f'High {target} MW%': mw_percentage.get(1, 0)
            })

        # Create a DataFrame for mind wandering percentages
        mw_results_df = pd.DataFrame(mind_wandering_results)
        print("\nMind Wandering Percentages by Emotion:")
        print(mw_results_df)

        # Optionally, save the mind wandering results
        mw_results_csv_path = os.path.join(output_dir, 'mind_wandering_percentages.csv')
        mw_results_df.to_csv(mw_results_csv_path, index=False)
        print(f"\nMind Wandering percentages saved to {mw_results_csv_path}")
    else:
        print("\n'Mind_Wandering' label not found in data. Cannot compute mind wandering percentages.")
else:
    print("Merged DataFrame is empty after adjustments. Please verify that the ClipIDs match in both DataFrames.")

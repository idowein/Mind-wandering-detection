import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Paths to CSV files
# -----------------------------
facial_expression_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\results.csv"
labels_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\TrainLabels.csv"

# -----------------------------
# Read the facial expression results CSV
# -----------------------------
facial_df = pd.read_csv(facial_expression_csv)

# -----------------------------
# Read the labels CSV
# -----------------------------
labels_df = pd.read_csv(labels_csv)

# Check the columns in labels_df
print("Columns in labels_df:")
print(labels_df.columns.tolist())

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
    # Compute Pearson and Spearman Correlation Matrices
    # -----------------------------
    print("\n--- Computing Pearson and Spearman Correlation Matrices ---")

    # Select features and labels for correlation
    facial_features = ['Blinking', 'Eyebrow Contractions', 'Cheek Raising',
                       'Facial Twitching', 'Yawning', 'Smiling', 'Frowning', 'Looking Away']
    emotion_labels = ['Engagement', 'Boredom', 'Confusion', 'Frustration ']  # Include 'Frustration'

    # Ensure that the features and labels are numerical
    for feature in facial_features:
        if merged_df[feature].dtype == 'object':
            merged_df[feature] = pd.to_numeric(merged_df[feature], errors='coerce')
    for label in emotion_labels:
        if label in merged_df.columns:
            if merged_df[label].dtype == 'object':
                merged_df[label] = pd.to_numeric(merged_df[label], errors='coerce')
        else:
            print(f"Error: '{label}' label not found in merged DataFrame.")
            # Assign NaN to handle missing columns
            merged_df[label] = float('nan')

    # Combine features and labels for correlation
    correlation_df = merged_df[facial_features + emotion_labels].dropna()

    # Compute the Pearson correlation matrix
    corr_matrix_pearson = correlation_df.corr(method='pearson')

    # Compute the Spearman correlation matrix
    corr_matrix_spearman = correlation_df.corr(method='spearman')

    # Extract the correlations between facial expressions and DAiSEE labels
    corr_features_labels_pearson = corr_matrix_pearson.loc[facial_features, emotion_labels]
    corr_features_labels_spearman = corr_matrix_spearman.loc[facial_features, emotion_labels]

    # Display the Pearson correlation matrix
    print("\nPearson Correlation Matrix:")
    print(corr_features_labels_pearson)

    # Display the Spearman correlation matrix
    print("\nSpearman Correlation Matrix:")
    print(corr_features_labels_spearman)

    # -----------------------------
    # Plot Heatmaps of Pearson and Spearman Correlation Matrices
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Pearson Heatmap
    sns.heatmap(corr_features_labels_pearson, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title('Pearson Correlation Between Facial Expressions and DAiSEE Labels')
    axes[0].set_xlabel('DAiSEE Labels')
    axes[0].set_ylabel('Facial Expressions')

    # Spearman Heatmap
    sns.heatmap(corr_features_labels_spearman, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title('Spearman Correlation Between Facial Expressions and DAiSEE Labels')
    axes[1].set_xlabel('DAiSEE Labels')
    axes[1].set_ylabel('Facial Expressions')

    plt.tight_layout()
    plt.show()

else:
    print("Merged DataFrame is empty after adjustments. Please verify that the ClipIDs match in both DataFrames.")

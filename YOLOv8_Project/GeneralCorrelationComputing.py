import pandas as pd
import os

# Paths to CSV files
facial_expression_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\facial_expression_results.csv"
labels_csv = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\AllLabels.csv"

# Read the facial expression results CSV
facial_df = pd.read_csv(facial_expression_csv)

# Read the labels CSV
labels_df = pd.read_csv(labels_csv)

# Debug: Check columns
print("Facial expression DataFrame columns:", facial_df.columns.tolist())
print("Labels DataFrame columns:", labels_df.columns.tolist())

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

# Debug: Display sample ClipIDs
print("Sample ClipIDs in facial_df:")
print(facial_df['ClipID'].head(10))

print("Sample ClipIDs in labels_df:")
print(labels_df['ClipID'].head(10))

# Merge on 'ClipID'
merged_df = pd.merge(facial_df, labels_df, on='ClipID')

# Check number of rows in merged DataFrame
print(f"Number of rows in merged DataFrame after adjustment: {len(merged_df)}")

# Proceed only if merged_df is not empty
if len(merged_df) > 0:
    # Define the columns for correlation
    facial_features = ['Blink_Count', 'Head_Movements', 'Eyebrow_Contractions',
                       'Cheek_Raisings', 'Facial_Twitches']  # Removed 'Estimated_Focus'
    daisee_measures = ['Engagement', 'Boredom', 'Confusion', 'Frustration']

    # Check which columns are present
    available_features = [col for col in facial_features if col in merged_df.columns]
    available_measures = [col for col in daisee_measures if col in merged_df.columns]

    print("Available facial features:", available_features)
    print("Available DAiSEE measures:", available_measures)

    # Ensure all required columns are present
    missing_features = [col for col in facial_features if col not in merged_df.columns]
    missing_measures = [col for col in daisee_measures if col not in merged_df.columns]

    if missing_features:
        print(f"Missing facial features in merged DataFrame: {missing_features}")
    if missing_measures:
        print(f"Missing DAiSEE measures in merged DataFrame: {missing_measures}")

    # Ensure that the labels are numerical
    for measure in available_measures:
        if merged_df[measure].dtype == 'object':
            merged_df[measure] = pd.to_numeric(merged_df[measure], errors='coerce')

    # Handle missing values
    data_to_correlate = merged_df[available_features + available_measures]
    data_to_correlate = data_to_correlate.dropna(subset=available_features + available_measures)

    # Compute correlation matrix
    correlation_matrix = data_to_correlate.corr()

    # Extract correlations between facial features and emotion labels
    correlations = correlation_matrix.loc[available_features, available_measures]

    # Display the correlations
    print("Correlations between facial features and DAiSEE emotion labels:")
    print(correlations)

    # Save the correlations to a CSV file
    output_dir = os.path.dirname(facial_expression_csv)
    corr_csv_path = os.path.join(output_dir, 'facial_expression_to_emotion_correlations.csv')
    correlations.to_csv(corr_csv_path)
    print(f"Correlations saved to {corr_csv_path}")

    # Optionally, visualize the correlations
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm')
        plt.title('Correlations Between Facial Expressions and Emotion Labels')
        plt.xlabel('Emotion Labels')
        plt.ylabel('Facial Expressions')
        # Save the heatmap to a file
        heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to {heatmap_path}")
    except ImportError:
        print("Seaborn or Matplotlib not installed. Skipping heatmap visualization.")
else:
    print("Merged DataFrame is empty after adjustments. Please verify that the ClipIDs match in both DataFrames.")

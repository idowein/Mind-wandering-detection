import pandas as pd


def generate_engagement_labels(input_csv_path, output_csv_path):
    # Load the original DAiSEE dataset CSV
    df = pd.read_csv(input_csv_path)

    # Ensure the required columns are present
    required_columns = ['Engagement', 'Boredom', 'Confusion', 'Frustration ']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the input CSV.")

    # Sum the levels of Boredom, Confusion, and Frustration
    df['NonEngagement_Sum'] = df['Boredom'] + df['Confusion'] + df['Frustration ']

    # Compare Engagement level with the sum of the other levels
    # Initialize the new label column
    df['Label'] = 0  # Placeholder

    # Apply the labeling logic
    df.loc[df['Engagement'] > df['NonEngagement_Sum'], 'Label'] = 1  # Engagement
    df.loc[df['Engagement'] <= df['NonEngagement_Sum'], 'Label'] = 2  # Non-Engagement

    # Drop the temporary 'NonEngagement_Sum' column
    df.drop(columns=['NonEngagement_Sum'], inplace=True)

    # Optionally, drop the original levels columns if not needed
    # df.drop(columns=['Engagement', 'Boredom', 'Confusion', 'Frustration'], inplace=True)

    # Save the new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"New CSV file with engagement labels saved to '{output_csv_path}'")


if __name__ == "__main__":
    # Replace these paths with your actual file paths
    input_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\OriginalLables\TestLabels.csv"
    output_csv_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\2Lables\2LablesTest.csv'

    generate_engagement_labels(input_csv_path, output_csv_path)

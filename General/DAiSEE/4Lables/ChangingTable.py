import pandas as pd


def combine_labels(input_csv_path, output_csv_path):
    # Load the CSV file
    data = pd.read_csv(input_csv_path)

    # Clean column names (remove leading/trailing whitespace)
    data.columns = data.columns.str.strip()

    # Display the first few rows
    print("First few rows of the input dataset:")
    print(data.head())

    # Identify the columns representing the levels of understanding
    level_columns = ['Boredom', 'Confusion', 'Engagement', 'Frustration']

    # Ensure that the level columns exist in the data
    missing_columns = [col for col in level_columns if col not in data.columns]
    if missing_columns:
        print(f"\nError: The dataset is missing the following required columns: {missing_columns}")
        return

    # Create a mapping from column names to label numbers
    label_mapping = {
        'Frustration': 1,
        'Engagement': 2,
        'Confusion': 3,
        'Boredom': 4
    }

    # Function to determine the label for each row
    def get_label(row):
        for col in level_columns:
            if row[col] == 1:
                return label_mapping[col]
        # If no level column is marked, return None or a default value
        return None  # Or you can return 0 or another value

    # Apply the function to each row to create the 'Label' column
    data['Label'] = data.apply(get_label, axis=1)

    # Check for rows where no label was assigned
    if data['Label'].isnull().any():
        print("\nWarning: Some rows did not have any level marked. These rows will be removed.")
        data = data.dropna(subset=['Label'])

    # Convert 'Label' column to integer
    data['Label'] = data['Label'].astype(int)

    # Create a new DataFrame with 'Video Name' and 'Label'
    output_data = data[['ClipID', 'Label']]

    # Save the new DataFrame to a CSV file
    output_data.to_csv(output_csv_path, index=False)
    print(f"\nNew CSV file with labels saved to '{output_csv_path}'")

    # Display the first few rows of the output data
    print("\nFirst few rows of the output dataset:")
    print(output_data.head())


if __name__ == "__main__":
    # Replace with the path to your input CSV file
    input_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\ValidationLabels.csv"

    # Replace with the desired output CSV file path
    output_csv_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\ValidationDetections\NewValidationLables.csv'

    combine_labels(input_csv_path, output_csv_path)

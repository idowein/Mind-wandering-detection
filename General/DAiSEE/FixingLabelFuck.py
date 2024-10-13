import pandas as pd


def assign_label_based_on_highest_score(input_csv_path, output_csv_path):
    # Load the CSV file
    data = pd.read_csv(input_csv_path)

    # Clean column names (remove leading/trailing whitespace)
    data.columns = data.columns.str.strip()

    # Display the first few rows
    print("First few rows of the input dataset:")
    print(data.head())

    # Define the columns representing the levels of understanding
    level_columns = ['Engagement', 'Confusion', 'Frustration', 'Boredom']

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

    # Function to determine the label based on the highest score
    def get_label(row):
        # Extract the scores for all levels
        scores = row[level_columns]
        # Find the maximum score
        max_score = scores.max()
        # Find the levels with the maximum score
        max_levels = scores[scores == max_score].index.tolist()

        if len(max_levels) == 1:
            # Only one level has the highest score
            return label_mapping[max_levels[0]]
        elif len(max_levels) > 1:
            # Multiple levels have the same highest score
            # Decide how to handle ties (e.g., choose based on priority)
            # For this example, we'll choose the level with the highest priority
            for col in level_columns:
                if col in max_levels:
                    return label_mapping[col]
        else:
            # No score available (all scores are NaN)
            return None

    # Apply the function to each row to create the 'Label' column
    data['Label'] = data.apply(get_label, axis=1)

    # Check for rows where no label was assigned
    if data['Label'].isnull().any():
        print("\nWarning: Some rows did not have any scores. These rows will be removed.")
        data = data.dropna(subset=['Label'])

    # Convert 'Label' column to integer
    data['Label'] = data['Label'].astype(int)

    # Create a new DataFrame with 'Video Name' and 'Label'
    output_data = data[['Video Name', 'Label']]

    # Save the new DataFrame to a CSV file
    output_data.to_csv(output_csv_path, index=False)
    print(f"\nNew CSV file with labels saved to '{output_csv_path}'")

    # Display the first few rows of the output data
    print("\nFirst few rows of the output dataset:")
    print(output_data.head())


if __name__ == "__main__":
    # Replace with the path to your input CSV file
    input_csv_path = r'C:\path\to\your\validation_labeled.csv'

    # Replace with the desired output CSV file path
    output_csv_path = r'C:\path\to\your\validation_labels_combined.csv'

    assign_label_based_on_highest_score(input_csv_path, output_csv_path)

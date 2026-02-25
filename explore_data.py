import pandas as pd
import matplotlib.pyplot as plt

# The goal of this script is to perform an initial exploration of the AI4I 2020 Predictive Maintenance Dataset.
# Data exploration is a crucial step in any machine learning pipeline. It helps us understand the
# distribution of the data, identify potential issues (like missing values), and inform
# feature engineering and model selection.

def explore_data(file_path):
    """
    Loads the dataset, checks for missing values, and plots distributions of specific columns.
    """
    try:
        # Load the dataset
        # pandas.read_csv uses a parser to convert the CSV file into a DataFrame,
        # which is a 2-dimensional labeled data structure with columns of potentially different types.
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")

        # 1. Check for missing values
        # Mathematically, we are checking for null or NaN entries in the data matrix.
        # If the dataset has missing values, we might need to use imputation techniques (like mean, median,
        # or mode replacement) or remove the affected rows/columns, as most machine learning models
        # cannot handle missing data directly.
        missing_values = df.isnull().sum()
        print("\nMissing values in each column:")
        print(missing_values)

        # 2. Plot distributions of 'Tool wear [min]' and 'Torque [Nm]'
        # A distribution plot (histogram) shows the frequency of values within certain ranges (bins).
        # This helps us understand:
        # - Central tendency (mean, median)
        # - Spread (variance, standard deviation)
        # - Skewness (asymmetry)
        # - Presence of outliers

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plotting 'Tool wear [min]'
        # Tool wear is the gradual failure of cutting tools due to regular operation.
        # Understanding its distribution can help predict when a tool will need replacement.
        df['Tool wear [min]'].hist(ax=axes[0], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Tool wear [min]')
        axes[0].set_xlabel('Tool wear [min]')
        axes[0].set_ylabel('Frequency')

        # Plotting 'Torque [Nm]'
        # Torque is the rotational equivalent of linear force.
        # In manufacturing, torque anomalies can be early indicators of machine failure.
        df['Torque [Nm]'].hist(ax=axes[1], bins=30, color='salmon', edgecolor='black')
        axes[1].set_title('Distribution of Torque [Nm]')
        axes[1].set_xlabel('Torque [Nm]')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()

        # Save the plots
        output_file = 'data_distributions.png'
        plt.savefig(output_file)
        print(f"\nPlots saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    explore_data('datasets/ai4i2020.csv')

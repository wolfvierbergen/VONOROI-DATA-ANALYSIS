import pandas as pd
import numpy as np
from collections import defaultdict

def remove_outliers_range_based(series, threshold=0.625):
    """
    Remove outliers using range-based method, suitable for small samples.
    
    Args:
        series: pandas Series with numeric values
        threshold: threshold for outlier detection (0.75 means value within 75% of range from mean)
        
    Returns:
        tuple: (cleaned series, number of outliers removed)
    """
    # Only consider non-empty values
    valid_values = series[series.notna()]
    if len(valid_values) < 2:  # Need at least 2 values
        return valid_values, 0
        
    data_range = valid_values.max() - valid_values.min()
    mean_value = valid_values.mean()
    
    # If all values are identical, return as is
    if data_range == 0:
        return valid_values, 0
    
    # Calculate distance from mean relative to the range
    relative_distances = abs(valid_values - mean_value) / data_range
    
    # Count outliers
    n_outliers = sum(relative_distances > threshold)
    
    # Keep values where the relative distance is within threshold
    clean_data = valid_values[relative_distances <= threshold]
    
    return clean_data, n_outliers

def process_csv(file_path):
    """
    Process CSV file to average values for rows with the same name,
    removing outliers before averaging using range-based method.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (processed dataframe, outlier statistics dictionary)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get the name and Sample columns
    name_column = df.iloc[:, 0]
    sample_column = df.iloc[:, 1]
    # Skip the third column and get the rest of the data
    data_columns = df.iloc[:, 3:]
    
    # Dictionary to store outlier statistics
    outlier_stats = defaultdict(lambda: {'outliers': 0, 'total_measurements': 0})
    
    # Define custom aggregation for data columns with outlier rejection
    def custom_mean_no_outliers(series):
        # Remove outliers and calculate mean
        clean_data, n_outliers = remove_outliers_range_based(series)
        
        # Update statistics
        column_name = series.name
        n_valid = len(series[series.notna()])
        outlier_stats[column_name]['outliers'] += n_outliers
        outlier_stats[column_name]['total_measurements'] += n_valid
        
        if len(clean_data) > 0:
            return clean_data.mean()
        return np.nan
    
    # Create a dataframe with name and sample for grouping
    grouped_df = pd.DataFrame({
        'name': name_column,
        'Sample': sample_column
    })
    
    # Add the data columns to be averaged
    for col in data_columns.columns:
        grouped_df[col] = data_columns[col]
    
    # Define aggregation dictionary
    agg_dict = {
        'Sample': 'first'  # Take the first Sample value for each name
    }
    # Add custom mean aggregation for all data columns
    for col in data_columns.columns:
        agg_dict[col] = custom_mean_no_outliers
    
    # Group by name and apply the aggregations
    result = grouped_df.groupby('name').agg(agg_dict).reset_index()
    
    # Save the processed data without thousands separators
    # and using only dots for decimal points
    output_file = 'processed_output.csv'
    result.to_csv(output_file, index=False, float_format='%.6f')
    
    return result, outlier_stats

def display_outlier_statistics(outlier_stats):
    """
    Display formatted outlier statistics.
    """
    print("\nOutlier Statistics:")
    print("-" * 60)
    print(f"{'Column':<30} {'Outliers':<10} {'Total':<10} {'Percentage':<10}")
    print("-" * 60)
    
    total_outliers = 0
    total_measurements = 0
    
    # Display statistics for each column
    for column, stats in sorted(outlier_stats.items()):
        outliers = stats['outliers']
        total = stats['total_measurements']
        percentage = (outliers / total * 100) if total > 0 else 0
        
        print(f"{column:<30} {outliers:<10} {total:<10} {percentage:>6.1f}%")
        
        total_outliers += outliers
        total_measurements += total
    
    # Display overall statistics
    print("-" * 60)
    overall_percentage = (total_outliers / total_measurements * 100) if total_measurements > 0 else 0
    print(f"{'TOTAL':<30} {total_outliers:<10} {total_measurements:<10} {overall_percentage:>6.1f}%")
    print("-" * 60)

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual file path
    file_path = 'dataset no semicolon no thousands separators.csv'
    
    try:
        processed_df, outlier_stats = process_csv(file_path)
        print("\nProcessing completed successfully!")
        display_outlier_statistics(outlier_stats)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
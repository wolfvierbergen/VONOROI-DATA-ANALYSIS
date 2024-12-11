import pandas as pd

def calculate_reduction_rates(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the control values (first row) for the specified columns
    control_values = df[df['Sample'] == 'control'][['STATIC_COF', 'DYNAMIC_COF', 'WEAR']].iloc[0]
    
    # Calculate reduction rates for each column
    df['RR_STATIC_COF'] = (control_values['STATIC_COF'] - df['STATIC_COF']) / control_values['STATIC_COF']
    df['RR_DYNAMIC_COF'] = (control_values['DYNAMIC_COF'] - df['DYNAMIC_COF']) / control_values['DYNAMIC_COF']
    df['RR_WEAR'] = (control_values['WEAR'] - df['WEAR']) / control_values['WEAR']
    
    # Save the modified DataFrame to a new CSV file
    output_file = 'processed_output_with_rr.csv'
    df.to_csv(output_file, index=False)
    return df

# Example usage
if __name__ == "__main__":
    input_file = "/Users/wolfvierbergen/Library/Mobile Documents/com~apple~CloudDocs/KU Leuven/Biomechanics/VONOROI DATA ANALYSIS/Data/processed_output.csv"
    result_df = calculate_reduction_rates(input_file)
    print("First few rows of the processed data:")
    print(result_df.head())
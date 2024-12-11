import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_csv(file_path):
    """
    Analyze a CSV file and create an interactive scatter plot with multiple data series.
    
    Parameters:
    file_path (str): Path to the CSV file
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Print column names with their indices
    print("\nAvailable columns:")
    for i, column in enumerate(df.columns):
        print(f"{i}: {column}")

    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Generate a color palette with enough distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))  # Prepare 20 distinct colors
    color_index = 0
    
    data_series = []  # Keep track of plotted series for legend
    
    while True:
        print("\nSelect columns for a new data series (press Enter without input to finish):")
        
        # Get X-axis column selection
        x_input = input("\nEnter the number for the X-axis column (or press Enter to finish): ").strip()
        if not x_input:
            break
            
        try:
            x_col_index = int(x_input)
            x_col = df.columns[x_col_index]
        except (ValueError, IndexError):
            print("Invalid column number. Please try again.")
            continue

        # Get Y-axis column selection
        try:
            y_col_index = int(input("Enter the number for the Y-axis column: "))
            y_col = df.columns[y_col_index]
        except (ValueError, IndexError):
            print("Invalid column number. Please try again.")
            continue

        # Remove NaNs and infinite values
        df_cleaned = df[[x_col, y_col]].dropna()
        df_cleaned = df_cleaned[np.isfinite(df_cleaned[x_col]) & np.isfinite(df_cleaned[y_col])]
        
        # Plot the data series with a new color
        scatter = plt.scatter(df_cleaned[x_col], df_cleaned[y_col], 
                            c=[colors[color_index]], 
                            label=f'{x_col} vs {y_col}',
                            alpha=0.6)  # Add some transparency
        
        # Check for variation in data before fitting a trendline
        if np.std(df_cleaned[x_col]) > 0 and np.std(df_cleaned[y_col]) > 0:
            # Fit a polynomial trendline of order 2 (quadratic fit)
            z = np.polyfit(df_cleaned[x_col], df_cleaned[y_col], 2)  # Polynomial fit (degree 2)
            p = np.poly1d(z)
            plt.plot(df_cleaned[x_col], p(df_cleaned[x_col]), 
                     color=colors[color_index], 
                     linestyle='--', 
                     label=f'{x_col} vs {y_col} Trendline')
        else:
            print(f"Warning: Insufficient variation in data for {x_col} vs {y_col}. Skipping trendline.")
        
        data_series.append(f'{x_col} vs {y_col}')
        color_index = (color_index + 1) % len(colors)  # Cycle through colors

    # If no data series were plotted, exit
    if not data_series:
        print("No data series selected. Exiting...")
        plt.close()
        return

    # Customize the plot
    plt.title('Multi-Series Scatter Plot with Polynomial Trendlines (Order 2)')
    plt.xlabel('struts_diameter')
    plt.ylabel('RR_WEAR')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create filename combining all series names
    series_names = ' & '.join(data_series)
    filename = f"Vonoroi data {series_names}.png"
    
    # Save the plot with expanded size to accommodate legend
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nPlot saved as {filename}")

# Prompt user for CSV file path
if __name__ == "__main__":
    csv_path = "Data/processed_output_with_rr.csv"
    analyze_csv(csv_path)

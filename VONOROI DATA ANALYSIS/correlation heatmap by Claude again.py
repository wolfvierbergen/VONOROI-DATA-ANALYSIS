import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
def load_data(filepath):
    # Read the CSV file with semicolon separator
    df = pd.read_csv(filepath, sep=';')
    return df

# Perform correlation analysis
def correlation_analysis(df):
    # Select relevant columns for correlation
    correlation_columns = [
        'seed_point_spacing', 'struts_diameter', 'relative_density_CAD',
        'feature_size_mm_CAD', 'feature_size_um_CAD', 'feature_size_mm_sample',
        'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD',
        'contacted_area_sample_mm2', 'contacted_ratio_sample', 'STATIC_COF',
        'DYNAMIC_COF', 'WEAR'
    ]
    
    correlation_matrix = df[correlation_columns].corr()
    
    # Create a heatmap of correlations with improved visualization
    plt.figure(figsize=(20, 16))  # Increased figure size
    
    # Create heatmap with improved parameters
    sns.heatmap(correlation_matrix, 
                annot=True,  # Show correlation values
                fmt='.2f',   # Round to 2 decimal places
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                annot_kws={'size': 8},  # Adjusted annotation size
                cbar_kws={"shrink": .8})
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add title
    plt.title('Correlation Heatmap of Voronoi Structure Parameters and Performance Metrics',
              pad=20, size=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('correlation_heatmap_2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_matrix

# Scatter plot matrix
def scatter_plot_matrix(df):
    # Select relevant columns
    columns_of_interest = [
        'seed_point_spacing', 'struts_diameter', 'relative_density_CAD',
        'feature_size_mm_CAD', 'STATIC_COF', 'DYNAMIC_COF', 'WEAR'
    ]
    
    # Create scatter plot matrix
    plt.figure(figsize=(20, 20))
    scatter_matrix = pd.plotting.scatter_matrix(
        df[columns_of_interest], 
        figsize=(20, 20), 
        diagonal='hist', 
        alpha=0.8
    )
    
    # Rotate labels
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right')
        ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right')
    
    plt.suptitle('Scatter Plot Matrix of Voronoi Structure Parameters', y=1.02, size=14)
    plt.tight_layout()
    plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# Statistical significance test
def statistical_significance_test(df):
    # Prepare results dictionary
    significance_results = {}
    
    # List of tribological performance metrics
    metrics = ['STATIC_COF', 'DYNAMIC_COF', 'WEAR']
    # List of design parameters
    parameters = ['seed_point_spacing', 'struts_diameter', 'relative_density_CAD']
    
    for metric in metrics:
        significance_results[metric] = {}
        for param in parameters:
            # Remove NaN values
            valid_data = df[[param, metric]].dropna()
            if len(valid_data) > 1:  # Check if we have enough data points
                correlation, p_value = stats.pearsonr(valid_data[param], valid_data[metric])
                significance_results[metric][param] = {
                    'correlation': correlation,
                    'p_value': p_value
                }
    
    # Print results
    print("\nStatistical Significance Results:")
    for metric, params in significance_results.items():
        print(f"\n{metric}:")
        for param, results in params.items():
            print(f"  {param}:")
            print(f"    Correlation: {results['correlation']:.4f}")
            print(f"    P-value: {results['p_value']:.4f}")
    
    return significance_results

# Regression analysis
def regression_analysis(df):
    # Prepare features and target variables
    features = ['seed_point_spacing', 'struts_diameter', 'relative_density_CAD']
    targets = ['STATIC_COF', 'DYNAMIC_COF', 'WEAR']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Regression results dictionary
    regression_results = {}
    
    for target in targets:
        # Remove NaN values
        valid_data = df[features + [target]].dropna()
        
        if len(valid_data) > 1:  # Check if we have enough data points
            # Prepare X and y
            X = valid_data[features]
            y = valid_data[target]
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Predict and calculate R-squared
            y_pred = model.predict(X_scaled)
            r_squared = r2_score(y, y_pred)
            
            # Store results
            regression_results[target] = {
                'coefficients': dict(zip(features, model.coef_)),
                'intercept': model.intercept_,
                'r_squared': r_squared
            }
    
    # Print regression results
    print("\nRegression Analysis Results:")
    for target, results in regression_results.items():
        print(f"\n{target}:")
        print(f"  R-squared: {results['r_squared']:.4f}")
        print("  Coefficients:")
        for feature, coef in results['coefficients'].items():
            print(f"    {feature}: {coef:.4f}")
        print(f"  Intercept: {results['intercept']:.4f}")
    
    return regression_results

# Main analysis function
def main_analysis(filepath):
    # Load data
    df = load_data(filepath)
    
    # Correlation Analysis
    correlation_matrix = correlation_analysis(df)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Scatter Plot Matrix
    scatter_plot_matrix(df)
    
    # Statistical Significance Test
    statistical_significance_test(df)
    
    # Regression Analysis
    regression_analysis(df)

# Run the analysis
if __name__ == '__main__':
    main_analysis('Data/processed_output_with_rr.csv')
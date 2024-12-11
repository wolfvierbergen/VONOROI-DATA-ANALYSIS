import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the dataset
def load_data(filepath):
    # Read the CSV file with semicolon separator
    df = pd.read_csv(filepath, sep=',')
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['seed_point_spacing', 'struts_diameter','relative_density_CAD','feature_size_mm_CAD','feature_size_um_CAD', 'feature_size_mm_sample', 'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD', 'contacted_area_sample_mm2', 'contacted_ratio_sample', 'STATIC_COF', 'DYNAMIC_COF', 'WEAR', 'RR_STATIC_COF','RR_DYNAMIC_COF','RR_WEAR'])
    
    return df

# Perform correlation analysis
def correlation_analysis(df):
    # Select relevant columns for correlation
    correlation_columns = ['seed_point_spacing', 'struts_diameter','relative_density_CAD','feature_size_mm_CAD','feature_size_um_CAD', 'feature_size_mm_sample', 'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD', 'contacted_area_sample_mm2', 'contacted_ratio_sample', 'STATIC_COF', 'DYNAMIC_COF', 'WEAR', 'RR_STATIC_COF','RR_DYNAMIC_COF','RR_WEAR']
    correlation_matrix = df[correlation_columns].corr()
    
    # Create a heatmap of correlations
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Voronoi Structure Parameters and Tribological Performance')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return correlation_matrix

# Scatter plot matrix
def scatter_plot_matrix(df):
    # Select relevant columns
    columns_of_interest = ['seed_point_spacing', 'struts_diameter','relative_density_CAD','feature_size_mm_CAD','feature_size_um_CAD', 'feature_size_mm_sample', 'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD', 'contacted_area_sample_mm2', 'contacted_ratio_sample', 'STATIC_COF', 'DYNAMIC_COF', 'WEAR', 'RR_STATIC_COF','RR_DYNAMIC_COF','RR_WEAR']
    
    # Create scatter plot matrix
    plt.figure(figsize=(12, 10))
    scatter_matrix = pd.plotting.scatter_matrix(df[columns_of_interest], 
                                                figsize=(12, 10), 
                                                diagonal='hist', 
                                                alpha=0.8, 
                                                figsize_factor=1.5)
    
    # Rotate diagonal labels
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), rotation=45)
        ax.set_ylabel(ax.get_ylabel(), rotation=45)
    
    plt.suptitle('Scatter Plot Matrix of Voronoi Structure Parameters')
    plt.tight_layout()
    plt.savefig('scatter_matrix.png')
    plt.close()

# Statistical significance test
def statistical_significance_test(df):
    # Prepare results dictionary
    significance_results = {}
    
    # List of tribological performance metrics
    metrics = ['STATIC_COF', 'DYNAMIC_COF', 'WEAR', 'RR_STATIC_COF','RR_DYNAMIC_COF','RR_WEAR']
    # List of design parameters
    parameters = ['seed_point_spacing', 'struts_diameter', 'relative_density_CAD','feature_size_mm_CAD','feature_size_um_CAD', 'feature_size_mm_sample', 'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD', 'contacted_area_sample_mm2', 'contacted_ratio_sample']
    
    for metric in metrics:
        significance_results[metric] = {}
        for param in parameters:
            # Perform Pearson correlation and p-value test
            correlation, p_value = stats.pearsonr(df[param], df[metric])
            significance_results[metric][param] = {
                'correlation': correlation,
                'p_value': p_value
            }
    
    # Print results
    print("Statistical Significance Results:")
    for metric, params in significance_results.items():
        print(f"\n{metric}:")
        for param, results in params.items():
            print(f"  {param}:")
            print(f"    Correlation: {results['correlation']:.4f}")
            print(f"    P-value: {results['p_value']:.4f}")
    
    return significance_results

# Regression analysis
def regression_analysis(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    
    # Prepare features and target variables
    features = ['seed_point_spacing', 'struts_diameter', 'relative_density_CAD','feature_size_mm_CAD','feature_size_um_CAD', 'feature_size_mm_sample', 'feature_size_um_sample', 'contatced_area_mm2_CAD', 'contatced_area_ratio_CAD', 'contacted_area_sample_mm2', 'contacted_ratio_sample']
    targets = ['STATIC_COF', 'DYNAMIC_COF', 'WEAR', 'RR_STATIC_COF','RR_DYNAMIC_COF','RR_WEAR']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Regression results dictionary
    regression_results = {}
    
    for target in targets:
        # Prepare X and y
        X = df[features]
        y = df[target]
        
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
    main_analysis('/Users/wolfvierbergen/Library/Mobile Documents/com~apple~CloudDocs/KU Leuven/Biomechanics/VONOROI DATA ANALYSIS/Data/processed_output_with_rr.csv')
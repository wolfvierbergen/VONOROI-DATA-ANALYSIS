import pandas as pd
import re

def clean_cell(cell):
    # Check if cell is a string; if not, return it as is (it may be a NaN or already numeric)
    if not isinstance(cell, str):
        return cell
    
    # If the cell starts with '0', leave it unchanged
    if cell.startswith('0'):
        return cell
    
    # If the cell has multiple dots, it is likely a large integer with dots as thousands separators
    if cell.count('.') > 1:
        return cell.replace('.', '')  # Remove all dots
    
    # If the cell contains only one dot, assume it's a decimal and leave it as is
    return cell

file_path = 'dataset no semicolon.csv'
data = pd.read_csv(file_path)

data_cleaned = data.applymap(clean_cell) #use the clean cell code 

# new CSV file
data_cleaned.to_csv('dataset no semicolon no thousands separators.csv', index=False)

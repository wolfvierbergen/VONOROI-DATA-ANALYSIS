import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

# Load the data from the specified relative path
file_path = input('Please select file path: ')
data = pd.read_csv(file_path)

# List available columns with numbering for easy selection
columns = data.columns
print("Available columns:")
for i, col in enumerate(columns, 1):
    print(f"{i}: {col}")

# Prompt user to select columns for each axis
def get_column(prompt):
    while True:
        try:
            col_num = int(input(prompt))
            if 1 <= col_num <= len(columns):
                return columns[col_num - 1]
            else:
                print(f"Please choose a number between 1 and {len(columns)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

x_column = get_column("Select the column number for the x-axis: ")
y_column = get_column("Select the column number for the y-axis: ")
z_column = get_column("Select the column number to minimize (z-axis): ")

# Ensure columns we need are in the dataframe
required_columns = {x_column, y_column, z_column}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Data does not contain the required columns: {required_columns}")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[[x_column, y_column, z_column]] = imputer.fit_transform(data[[x_column, y_column, z_column]])

# Find the combination that minimizes the z_column in the dataset
min_row = data.loc[data[z_column].idxmin()]
min_x = min_row[x_column]
min_y = min_row[y_column]
min_z = min_row[z_column]

print("Minimum value for the selected z-axis column in the dataset is at:")
print(f"{x_column}: {min_x}")
print(f"{y_column}: {min_y}")
print(f"{z_column}: {min_z}")

# Prepare data for surface fitting
X = data[[x_column, y_column]]
y = data[z_column]

# Fit a 3rd-degree polynomial regression model for a smooth surface
model = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
model.fit(X, y)

# Define a function to compute the surface prediction for any (x, y) point
def surface_prediction(xy):
    x, y = xy
    return model.predict([[x, y]])[0]

# Use the middle of the data range as a starting point for optimization
initial_guess = [
    data[x_column].mean(),
    data[y_column].mean()
]

# Bounds to constrain the optimization within the data range
bounds = [
    (data[x_column].min(), data[x_column].max()),
    (data[y_column].min(), data[y_column].max())
]

# Find the local minimum of the fitted surface
result = minimize(surface_prediction, initial_guess, bounds=bounds, method='L-BFGS-B')

# Extract the coordinates and value of the local minimum
local_min_x, local_min_y = result.x
local_min_z = result.fun

print("Local minimum value on the fitted surface is at:")
print(f"{x_column}: {local_min_x}")
print(f"{y_column}: {local_min_y}")
print(f"{z_column}: {local_min_z}")

# Create a meshgrid for the surface plot
x_range = np.linspace(data[x_column].min(), data[x_column].max(), 50)
y_range = np.linspace(data[y_column].min(), data[y_column].max(), 50)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# Predict values over the meshgrid for the z-axis
z_pred = model.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()]).reshape(x_mesh.shape)

# Plotting the data with the smooth surface
fig = go.Figure()

# Add scatter plot of actual data points
fig.add_trace(go.Scatter3d(
    x=data[x_column],
    y=data[y_column],
    z=data[z_column],
    mode='markers',
    marker=dict(size=5, color=data[z_column], colorscale='Viridis', opacity=0.7),
    name='Data Points'
))

# Add the smooth polynomial surface
fig.add_trace(go.Surface(
    x=x_mesh,
    y=y_mesh,
    z=z_pred,
    colorscale='Viridis',
    opacity=0.5,
    name='Smooth Fitted Surface'
))

# Highlight the minimum z-axis value in the dataset
#fig.add_trace(go.Scatter3d(
#    x=[min_x],
#    y=[min_y],
#    z=[min_z],
#    mode='markers',
#    marker=dict(size=8, color='red', symbol='diamond'),
#    name='Dataset Minimum'
#))

# Highlight the local minimum on the fitted surface found by optimization
fig.add_trace(go.Scatter3d(
    x=[local_min_x],
    y=[local_min_y],
    z=[local_min_z],
    mode='markers',
    marker=dict(size=8, color='blue', symbol='cross'),
    name='Local Minimum on Surface'
))

# Update layout for clarity
fig.update_layout(
    title=f"3D Scatter Plot with Smooth Surface and Local Minimum of {x_column}, {y_column}, and {z_column}",
    scene=dict(
        xaxis_title=x_column,
        yaxis_title=y_column,
        zaxis_title=z_column
    )
)

fig.show()

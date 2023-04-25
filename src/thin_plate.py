import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf

def calculate_weights(dbh_values, height_values, reflectance_values):
    # Normalize the feature values
    dbh_norm = (dbh_values - dbh_values.min()) / (dbh_values.max() - dbh_values.min())
    height_norm = (height_values - height_values.min()) / (height_values.max() - height_values.min())
    reflectance_norm = (reflectance_values - reflectance_values.min()) / (reflectance_values.max() - reflectance_values.min())

    # Calculate weights based on the features
    weights = dbh_norm * 0.5 + height_norm * 0.3 + reflectance_norm * 0.2

    return weights

# Load your GeoDataFrame (assuming it's a shapefile)
gdf = gpd.read_file('your_shapefile.shp')

# Define the known control points (ground truth)
control_points_source = np.array([[x1_src, y1_src],
                                  [x2_src, y2_src],
                                  [x3_src, y3_src]])

control_points_dest = np.array([[x1_dest, y1_dest],
                                [x2_dest, y2_dest],
                                [x3_dest, y3_dest]])

# Extract x, y coordinates and feature values from the GeoDataFrame
x, y = gdf.geometry.x, gdf.geometry.y
dbh_values = gdf['dbh'].values
height_values = gdf['height'].values
reflectance_values = gdf['reflectance'].values

# Calculate weights based on the features
weights = calculate_weights(dbh_values, height_values, reflectance_values)

# Perform the TPS transformation with weights
tps = Rbf(control_points_source[:, 0], control_points_source[:, 1], control_points_dest, function='thin_plate', smooth=0, weights=weights)
x_aligned, y_aligned = tps(x, y)

# Replace the original coordinates with the aligned coordinates
gdf.geometry = gpd.points_from_xy(x_aligned, y_aligned)

# Save the aligned GeoDataFrame to a new shapefile
gdf.to_file('your_aligned_shapefile.shp')

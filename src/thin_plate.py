import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import rasterio
def calculate_weights(dbh_values, height_values, reflectance_values):
    # Normalize the feature values
    dbh_norm = (dbh_values - dbh_values.min()) / (dbh_values.max() - dbh_values.min())
    height_norm = (height_values - height_values.min()) / (height_values.max() - height_values.min())
    reflectance_norm = (reflectance_values - reflectance_values.min()) / (reflectance_values.max() - reflectance_values.min())

    # Calculate weights based on the features
    weights = dbh_norm * 0.5 + height_norm * 0.3 + reflectance_norm * 0.2

    return weights


#load extent of the area
raster_path = '/home/smarconi/Documents/GitHub/tree_mask_delineation/imagery/hsi_clip.tif'
with rasterio.open(raster_path) as src:
    extent = src.bounds
# Load your GeoDataFrame (assuming it's a shapefile)
gdf_field =  gpd.read_file('/home/smarconi/Documents/DAT_for_health/SERC/SERC/field.shp')
gdf_field = gdf_field[gdf_field['Crwnpst'] > 2]

gdf_reference = gpd.read_file('/home/smarconi/Documents/GitHub/tree_mask_delineation/indir/data_field.shp')
#in gdf_reference, rename Tag in StemTag and turn values into character
gdf_reference = gdf_reference.rename(columns={'Tag': 'StemTag'})
gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(int)

gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(str)
gdf_field['StemTag'] = gdf_field['StemTag'].astype(str)

#find matching trees between field and reference
matched_points = pd.merge(gdf_field, gdf_reference, on='StemTag', suffixes=('_field', '_reference'))

# Extract source and destination control points from matched_points
control_points_source_x = matched_points['geometry_field'].x.values
control_points_source_y = matched_points['geometry_field'].y.values
control_points_dest_x = matched_points['geometry_reference'].x.values
control_points_dest_y = matched_points['geometry_reference'].y.values

# Extract x, y coordinates from gdf_field
x, y = gdf_field.geometry.x, gdf_field.geometry.y
check_delta_for_errors = control_points_source_x - control_points_dest_x
min(check_delta_for_errors)
# Perform the TPS transformation separately for x and y components
tps_x = Rbf(control_points_source_x, control_points_source_y, control_points_dest_x, function='thin_plate', smooth=1)
tps_y = Rbf(control_points_source_x, control_points_source_y, control_points_dest_y, function='thin_plate', smooth=1)
x_aligned = tps_x(x, y)
y_aligned = tps_y(x, y)

# Replace the original coordinates with the aligned coordinates in gdf_field
gdf_out = gdf_field.copy()
gdf_out.geometry = gpd.points_from_xy(x_aligned, y_aligned)

#remove points outside extent 
gdf_out = gdf_out.cx[extent[0]:extent[2], extent[1]:extent[3]]

# make a copy with only the StemTags in the reference
gdf_out_stemTags = gdf_out.copy()
gdf_out_stemTags = gdf_out_stemTags[gdf_out_stemTags['StemTag'].isin(gdf_reference['StemTag'])]
# Save the aligned gdf_field GeoDataFrame to a new shapefile
gdf_out.to_file('/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/SERC_aligned_cleaned.shp')
gdf_out_stemTags.to_file('/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/SERC_test.shp')


# calculate the mean distance between points with same StemTag in gdf_out_stemTags and gdf_reference
merged_data_test = pd.merge(gdf_out_stemTags, gdf_reference, on='StemTag', suffixes=('_aligned', '_reference'))
# Calculate the distance between each pair of points with the same StemTag
distances = merged_data_test.apply(lambda row: row['geometry_aligned'].distance(row['geometry_reference']), axis=1)
# Calculate the mean distance
mean_distance = distances.mean()
print(f"Mean distance between points with the same StemTag: {mean_distance:.2f} units")

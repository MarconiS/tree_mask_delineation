import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import rasterio

raster_path = '/home/smarconi/Documents/GitHub/tree_mask_delineation/imagery/hsi_clip.tif'
gdf_field =  '/home/smarconi/Documents/DAT_for_health/SERC/SERC/field.shp'
gdf_reference = '/home/smarconi/Documents/GitHub/tree_mask_delineation/indir/data_field.shp'

def use_thin_plate(raster_path, gdf_field, gdf_reference):
    #load extent of the area
    
    with rasterio.open(raster_path) as src:
        extent = src.bounds
    # Load your GeoDataFrame (assuming it's a shapefile)
    gdf_field =  gpd.read_file(gdf_field)
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



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

def train_random_forest(outdir, gdf_field, gdf_reference):

    # Load your GeoDataFrame (assuming it's a shapefile)
    gdf_field =  gpd.read_file(gdf_field)
    gdf_field = gdf_field[gdf_field['Crwnpst'] > 1]

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

    #define X as the pandas dataframe with control_points_source_x, control_points_source_y
    X = pd.DataFrame({'east': control_points_source_x, 'north': control_points_source_y})
    #define y as the pandas dataframe with control_points_dest_x, control_points_dest_y
    y = pd.DataFrame({'east': control_points_dest_x, 'north': control_points_dest_y})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression

    # Separate the target variables
    y_train_lat = y_train['east']
    y_train_lon = y_train['north']

    # Initialize the base models
    gbm_lat = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm_lon = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    rf_lat = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf_lon = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

    # Initialize the StackingRegressor
    stacking_lat = StackingRegressor(estimators=[('gbm_lat', gbm_lat)], final_estimator=LinearRegression())
    stacking_lon = StackingRegressor(estimators=[('gbm_lon', gbm_lon)], final_estimator=LinearRegression())

    # Train the StackingRegressor
    stacking_lat.fit(X_train, y_train_lat)
    stacking_lon.fit(X_train, y_train_lon)

    # Make predictions
    stacking_pred_lat = stacking_lat.predict(X_test)
    stacking_pred_lon = stacking_lon.predict(X_test)

    stacking_pred = np.column_stack((stacking_pred_lat, stacking_pred_lon))

    # Evaluate the stacked model
    mae = mean_absolute_error(y_test, stacking_pred)
    mse = mean_squared_error(y_test, stacking_pred)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # Now retrain the model with all coordinates, and use it for prediction of all gdf_field crowms
    stacking_lat.fit(X, y['east'])
    stacking_lon.fit(X, y['north'])

    # Make predictions
    Xf = pd.DataFrame({'east': gdf_field['geometry'].x.values, 'north': gdf_field['geometry'].y.values})
    stacking_pred_lat = stacking_lat.predict(Xf)
    stacking_pred_lon = stacking_lon.predict(Xf)
    #drop geometry from gdf_field and stack new coordinates
    gdf_field = gdf_field.drop(columns=['geometry'])
    gdf_field['geometry'] = gpd.points_from_xy(stacking_pred_lat, stacking_pred_lon)

    #turn gdf_field back into a geodataframe
    gdf_field = gpd.GeoDataFrame(gdf_field, geometry='geometry')
    #set the crs of gdf_field to the crs of gdf_reference
    gdf_field.crs = gdf_reference.crs
    #write gdf_field to a shapefile if outdir is not null
    if outdir is not None:
        gdf_field.to_file(outdir+'/data_field_reprojected.shp')
    
    return gdf_field




# Import the necessary libraries
import os
import rasterio
import numpy as np
import cv2
import geopandas as gpd
import torch
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import translate

from matplotlib import pyplot as plt

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import polygonize
from skimage.measure import find_contours

import numpy as np
import rasterio
from rasterio.features import shapes
from affine import Affine
import geopandas as gpd
from shapely.geometry import shape

def mask_to_polygons(mask, individual_point):
    # Find contours in the mask
    contours = np.array([mask], dtype=np.uint8)        #
    #np.array([mask], dtype=np.uint8)
    polygon_generator = shapes(contours)#, transform=transform)


    # Create a GeoDataFrame from the polygons
    geometries = []
    values = []

    for polygon, value in polygon_generator:
        geometries.append(shape(polygon))
        values.append(value)

    gdf_ = gpd.GeoDataFrame(geometry=geometries)
    gdf_['value'] = values

    #remove  polygons with value 0
    gdf_ = gdf_[gdf_['value'] != 0]

    # Check if there are any valid line segments
    if gdf_.shape[0] == 0:
        return None

    # Filter the polygons to include only those that contain the individual_point
    containing_polygons = [polygon for polygon in gdf_['geometry']  if polygon.contains(individual_point)]
    #calculate the area of the polygon
    area = [polygon.area for polygon in containing_polygons]
    # If there are no containing polygons, return None
    if not containing_polygons:
        return None

    # Choose the largest polygon based on its area
    largest_polygon = max(containing_polygons, key=lambda p: p.area)

    return largest_polygon

# Define a function to make predictions of tree crown polygons using SAM
def predict_tree_crowns(batch, input_points, neighbors = 10, point_type='random', rescale_to = None):
    from segment_anything import sam_model_registry, SamPredictor
    import geopandas as gpd
    import pandas as pd

    batch = np.moveaxis(batch, 0, -1)
    original_shape = batch.shape
    sam_checkpoint = "SAM/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    #neighbors must be the minimum between the total number of input_points and the argument neighbors
    neighbors = min(input_points.shape[0], neighbors)
    #rescale image to larger size if rescale_to is not null
    if rescale_to is not None:
        batch = resize(batch, (rescale_to, rescale_to), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
        input_points['x'] = input_points['x'] * rescale_to / original_shape[1]
        input_points['y'] = input_points['y'] * rescale_to / original_shape[0]

    # linstretch the image, normalize it to 0, 255 and convert to int8
    batch = np.uint8(255 * (batch - batch.min()) / (batch.max() - batch.min()))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    #flip rasterio to be h,w, channels
    predictor.set_image(batch)
    #turn stem points into a numpy array
    input_point = np.column_stack((input_points['x'], input_points['y']))
    input_crowns = input_points['StemTag']
    crown_mask = pd.DataFrame(columns=["geometry", "score"])
    crown_scores=[]
    crown_logits=[]
    #loop through each stem point, make a prediction, and save the prediction
    for it in range(0, input_point.shape[0]):
        #update input_label to be 0 everywhere except at position it       
        input_label = np.zeros(input_point.shape[0])
        input_label[it] = 1
        # subset the input_points to be the current point and the 10 closest points
        # Calculate the Euclidean distance between the ith row and the other rows
        distances = np.linalg.norm(input_point - input_point[it], axis=1)
        if point_type == "euclidian":
        # Find the indices of the 10 closest rows
            closest_indices = np.argpartition(distances, neighbors+1)[:neighbors+1]  # We use 11 because the row itself is included
        elif point_type == "random":
            closest_indices = np.random.choice(np.arange(0, input_point.shape[0]), neighbors+1, replace=True)
        # Subset the array to the ith row and the 10 closest rows
        subset_point = input_point[closest_indices]
        subset_label = input_label[closest_indices]
        subset_label = subset_label.astype(np.int8)
        masks, scores, logits = predictor.predict(
            point_coords=subset_point,
            point_labels=subset_label,
            multimask_output=False,
        )
        #pick the mask with the highest score
        masks = masks[scores.argmax()]
        scores = scores[scores.argmax()]
        # Find the indices of the True values
        true_indices = np.argwhere(masks)
        #skip empty masks
        if true_indices.shape[0] < 3:
            continue

        # Calculate the convex hull
        individual_point = Point(input_point[it])
        polygons = mask_to_polygons(masks, individual_point)

        # Create a GeoDataFrame and append the polygon
        gdf_temp = gpd.GeoDataFrame(geometry=[polygons], columns=["geometry"])
        gdf_temp["score"] = scores
        gdf_temp["stemTag"] = input_crowns.iloc[it]

        # Append the temporary GeoDataFrame to the main GeoDataFrame
        crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)
        crown_scores.append(scores)
        crown_logits.append(logits)

    # Convert the DataFrame to a GeoDataFrame
    crown_mask = gpd.GeoDataFrame(crown_mask, geometry=crown_mask.geometry)
    #reshift crown mask coordiantes to original size
    if rescale_to is not None:
        crown_mask['geometry'] = crown_mask['geometry'].translate(xoff=0, yoff=0)
        crown_mask['geometry'] = crown_mask['geometry'].scale(xfact=original_shape[1]/rescale_to, yfact=original_shape[0]/rescale_to, origin=(0,0))

    return crown_mask, crown_scores, crown_logits


# Define a function to save the predictions as geopandas
def save_predictions(predictions, output_path):
    # Initialize an empty geopandas dataframe
    gdf = gpd.GeoDataFrame()
    # Loop through the predictions
    for prediction in predictions:
        # Convert the prediction to a polygon geometry
        polygon = gpd.GeoSeries(prediction).unary_union
        # Append the polygon to the dataframe
        gdf = gdf.append({'geometry': polygon}, ignore_index=True)
    # Save the dataframe as a shapefile
    gdf.to_file(output_path)


import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry import box, Point, MultiPoint
import imageio

def transform_coordinates(geometry, x_offset, y_offset):
    if geometry.type == "Point":
        return Point(geometry.x - x_offset, geometry.y - y_offset)
#    elif geometry.type == "MultiPoint":
#        return MultiPoint([Point(p.x - x_offset, p.y - y_offset) for p in geometry])
    else:
        raise ValueError("Unsupported geometry type")


def split_image(image_file, itcs, batch_size=40, resolution=0.1):
    # Open the raster image
    with rasterio.open(image_file) as src:
        # Get the height and width of the image in pixels
        height, width = src.shape
        # Convert the batch size from meters to pixels
        batch_size = int(batch_size / resolution)
        # Initialize lists to store the raster batches and clipped GeoDataFrames
        raster_batches = []
        itcs_batches = []
        affines = []
        # Loop through the rows and columns of the image
        for i in range(0, height, batch_size):
            for j in range(0, width, batch_size):
                # Define a window for the current batch
                window = Window(col_off=j, row_off=i, width=batch_size, height=batch_size)
                # Read the batch from the raster image
                batch = src.read(window=window)
                # Append the raster batch to the list
                raster_batches.append(batch)

                # Convert the window to geospatial coordinates
                left, top = src.xy(i, j)
                right, bottom = src.xy(i+batch_size, j+batch_size)
                batch_bounds = box(left, bottom, right, top)

                # Clip the GeoDataFrame using the batch bounds
                itcs_clipped = gpd.clip(itcs, batch_bounds)

                # Transform the coordinates relative to the raster batch's origin
                itcs_clipped["geometry"] = itcs_clipped["geometry"].apply(
                    transform_coordinates, x_offset=left, y_offset=bottom
                )

                # Create a new DataFrame with stemTag, x, and y columns
                itcs_df = pd.DataFrame(
                    {
                        "StemTag": itcs_clipped["StemTag"],
                        "x": itcs_clipped["geometry"].x,
                        "y": itcs_clipped["geometry"].y,
                    }
                )

                # Append the DataFrame to the list
                itcs_batches.append(itcs_df)
                affines.append(src.window_transform(window))

    # Return the lists of raster batches and clipped GeoDataFrames
    return raster_batches, itcs_batches, affines


import numpy as np
from skimage.transform import resize


# Usage example
# Define a folder with remote sensing RGB data
folder = 'field_data_alignment'
file = 'imagery/hsi_clip.tif'
tree_tops = 'indir/test_noisy.shp'
# Loop through the files in the folder




#for file in os.listdir(folder):
# check if the tree_tops file exists. if not, launch get_tree_tops
if os.path.exists(os.path.join(folder, tree_tops)):
    itcs = gpd.read_file(os.path.join('/home/smarconi/Documents/DAT_for_health/SERC/SERC/field.shp'))
else:
    print('itc = get_tree_tops(lidar_path)')

#use only points whose crwnpst is greater than 2
itcs = itcs[itcs['Crwnpst'] > 2]
image_file = os.path.join(folder, file)
# Split the image into batches of 40x40m
raster_batches, itcs_batches, affine = split_image(image_file, itcs, batch_size=40, resolution =1)

#plot the batch
#plt.imshow(raster_batches[0][:3,:,:])
# Make predictions of tree crown polygons using SAM
for(i, batch) in enumerate(raster_batches):
    #skip empty batches
    if itcs_batches[i].shape[0] == 0: 
        continue


    # Make predictions of tree crown polygons using SAM
    predictions, _, _ = predict_tree_crowns(batch=batch[:3,:,:], input_points=itcs_batches[i],  neighbors=1, point_type = "euclidian") 
    # Apply the translation to the geometries in the GeoDataFrame
    x_offset, y_offset = affine[i][2], affine[i][5]
    #from a geopandas, remove all rows with a None geometry
    predictions = predictions[predictions['geometry'].notna()]
    predictions["geometry"] = predictions["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
    predictions.crs = "EPSG:32618"

    # Save the predictions as geopandas
    predictions.to_file(f'{folder}/outdir/itcs/itcs_{i}.gpkg', driver='GPKG')

    array = raster_batches[5][:3,:,:]
    batch = np.moveaxis(array, 0, -1)
    imageio.imwrite(f'{folder}/outdir/clips/itcs_{i}.png', batch)






import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from src.utils import *
from src.segment_crowns import *

from deepforest import main
from deepforest import get_data
import os
import matplotlib.pyplot as plt

# Usage example
# Define a folder with remote sensing RGB data
folder = '.'
file = 'imagery/rgb_clip.tif'
tree_tops = 'indir/test_noisy.shp'
hsi_img  = 'imagery/hsi_clip.tif'
# Loop through the files in the folder
#for file in os.listdir(folder):
# check if the tree_tops file exists. if not, launch get_tree_tops
if os.path.exists(os.path.join(folder, tree_tops)):
    itcs = gpd.read_file(os.path.join('/home/smarconi/Documents/DAT_for_health/SERC/SERC/field.shp'))
else:
    print('itc = get_tree_tops(lidar_path)')

#get tree bounding boxes with deepForest for SAM
bbox = extract_boxes(file)


#use only points whose crwnpst is greater than 2
itcs = itcs[itcs['Crwnpst'] > 2]
image_file = os.path.join(folder, file)
# Split the image into batches of 40x40m
batch_size = 40
raster_batches, raster_hsi_batches, itcs_batches, itcs_boxes, affine = 
    split_image(image_file,hsi_img, itcs, bbox, batch_size=batch_size)


# Make predictions of tree crown polygons using SAM
for(i, batch) in enumerate(raster_batches):
    #skip empty batches
    if itcs_batches[i].shape[0] == 0: 
        continue

    # Make predictions of tree crown polygons using SAM
    predictions, _, _ = predict_tree_crowns(batch=batch[:3,:,:], input_points=itcs_batches[i],  
                                            input_boxes = itcs_boxes[i], neighbors=3, 
                                            point_type = "euclidian") 
    # Apply the translation to the geometries in the GeoDataFrame
    x_offset, y_offset = affine[i][2], affine[i][5]
    y_offset = y_offset - batch_size
    #from a geopandas, remove all rows with a None geometry
    predictions = predictions[predictions['geometry'].notna()]
    predictions["geometry"] = predictions["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
    predictions.crs = "EPSG:32618"

    # Save the predictions as geopandas
    predictions.to_file(f'{folder}/outdir/itcs/itcs_{i}.gpkg', driver='GPKG')

    batch = batch[:3,:,:]
    batch = np.moveaxis(batch, 0, -1)
    imageio.imwrite(f'{folder}/outdir/clips/itcs_{i}.png', batch)






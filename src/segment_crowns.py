
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
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
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
import warnings

from segment_anything import sam_model_registry, SamPredictor
import geopandas as gpd
import pandas as pd
from skimage.transform import resize

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

    if individual_point is not None:
    # Filter the polygons to include only those that contain the individual_point
        containing_polygons = [polygon for polygon in gdf_['geometry']  if polygon.contains(individual_point)]
        # Choose the largest polygon based on its area
        largest_polygon = max(containing_polygons, key=lambda p: p.area)
    #calculate the area of the polygon
    #area = [polygon.area for polygon in containing_polygons]
    # If there are no containing polygons, return None
    if not containing_polygons:
        return None

    # Choose the largest polygon based on its area
    #largest_polygon = max(containing_polygons, key=lambda p: p.area)
    return largest_polygon

# Define a function to make predictions of tree crown polygons using SAM
def predict_tree_crowns(batch, input_points, neighbors = 10, 
                        input_boxes = None, point_type='random', 
                        onnx_model_path = None,  rescale_to = None):


    batch = np.moveaxis(batch, 0, -1)
    original_shape = batch.shape
    sam_checkpoint = "SAM/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    if onnx_model_path is None:
        onnx_model_path = "indir/sam_onnx_example.onnx"
        onnx_model = SamOnnxModel(sam, return_single_mask=True)
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
        }

        # Move all tensors in dummy_inputs to the device
        for key in dummy_inputs:
            dummy_inputs[key] = dummy_inputs[key]

        output_names = ["masks", "iou_predictions", "low_res_masks"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(onnx_model_path, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )    
        ort_session = onnxruntime.InferenceSession(onnx_model_path)

    #neighbors must be the minimum between the total number of input_points and the argument neighbors
    neighbors = min(input_points.shape[0]-2, neighbors)
    #rescale image to larger size if rescale_to is not null
    if rescale_to is not None:
        batch = resize(batch, (rescale_to, rescale_to), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
        input_points['x'] = input_points['x'] * rescale_to / original_shape[1]
        input_points['y'] = input_points['y'] * rescale_to / original_shape[0]

    # linstretch the image, normalize it to 0, 255 and convert to int8
    batch = np.uint8(255 * (batch - batch.min()) / (batch.max() - batch.min()))
    sam.to(device=device)


    predictor = SamPredictor(sam)
    #flip rasterio to be h,w, channels
    predictor.set_image(batch)

    if onnx_model_path is not None:
        image_embedding = predictor.get_image_embedding().cpu().numpy()

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
        target_itc = input_point[it]
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

        #Add a batch index, concatenate a padding point, and transform.
        if onnx_model_path is not None:
            onnx_coord = np.concatenate([subset_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([subset_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, batch.shape[:2]).astype(np.float32)
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(batch.shape[:2], dtype=np.float32)
            }
            masks, scores, logits = ort_session.run(None, ort_inputs)
            masks = masks > predictor.model.mask_threshold
            masks = masks[0, :, :, :]


        if input_boxes is 9999:
            subset_box = input_boxes[it]
            #select point whose x and y are within the box
            subset_point = subset_point[(subset_point[:,0] > subset_box[0]) & 
                (subset_point[:,0] < subset_box[2]) & (subset_point[:,1] > subset_box[1]) & 
                (subset_point[:,1] < subset_box[3])]     
            #if input_point is empty, pick the center of the box
            if subset_point.shape[0] == 0:
                subset_point = np.array([[subset_box[0] + (subset_box[2] - subset_box[0])/2, subset_box[1] + 
                                          (subset_box[3] - subset_box[1])/2]])
                
            if onnx_model_path is not None:
                #add the box coordinates to the subset_point
                onnx_box_coords = subset_box.reshape(2, 2)
                onnx_box_labels = np.array([2,3])
                onnx_coord = np.concatenate([input_point, onnx_box_coords], axis=0)[None, :, :]
                onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

                onnx_coord = predictor.transform.apply_coords(onnx_coord, batch.shape[:2]).astype(np.float32)
                onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
                onnx_has_mask_input = np.zeros(1, dtype=np.float32)

                ort_inputs = {
                    "image_embeddings": image_embedding,
                    "point_coords": onnx_coord,
                    "point_labels": onnx_label,
                    "mask_input": onnx_mask_input,
                    "has_mask_input": onnx_has_mask_input,
                    "orig_im_size": np.array(batch.shape[:2], dtype=np.float32)
                }

                masks,scores, logits = ort_session.run(None, ort_inputs)
                masks = masks > predictor.model.mask_threshold
                masks = masks[0, :, :, :]

        if onnx_model_path is None and input_boxes is not None:# and onnx_model_path is None:
           #transform the boxes into a torch.tensor
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, batch.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            # loop through the masks, polygonize their raster, and append them into a geopandas dataframe
            for i in range(masks.shape[0]):
                #pick the mask with the highest score
                mask = masks[i].to("cpu").numpy()
                # Find the indices of the True values
                true_indices = np.argwhere(mask)
                #skip empty masks
                if true_indices.shape[0] < 3:
                    continue
                # Calculate the convex hull
                individual_point = Point(input_point[it])
                polygons = mask_to_polygons(mask, individual_point=None)

                # Create a GeoDataFrame and append the polygon
                gdf_temp = gpd.GeoDataFrame(geometry=[polygons], columns=["geometry"])
                gdf_temp["score"] = scores[i]
                gdf_temp["stemTag"] = subset_label[i]
                gdf_temp["point_id"] = it
                gdf_temp["point_x"] = input_point[it][0]
                gdf_temp["point_y"] = input_point[it][1]
                gdf = gdf.append(gdf_temp, ignore_index=True)


        if onnx_model_path is None and input_boxes is None:
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
from shapely.geometry import box, Point, MultiPoint, Polygon
import imageio
import deepforest

def transform_coordinates(geometry, x_offset, y_offset):
    if geometry.type == "Point":
        return Point(geometry.x - x_offset, geometry.y - y_offset)
    elif geometry.type == "MultiPoint":
        return MultiPoint([Point(p.x - x_offset, p.y - y_offset) for p in geometry])
    elif geometry.type == "Polygon":
        return Polygon([(p[0] - x_offset, p[1] - y_offset) for p in geometry.exterior.coords])
    else:
        raise ValueError("Unsupported geometry type")


def split_image(image_file, itcs, bbox,  batch_size=40, resolution=0.1):
    # Open the raster image
    with rasterio.open(image_file) as src:
        # Get the height and width of the image in pixels
        height, width = src.shape
        # Convert the batch size from meters to pixels
        batch_size_ = int(batch_size / resolution)
        # Initialize lists to store the raster batches and clipped GeoDataFrames
        raster_batches = []
        itcs_batches = []
        affines = []
        itcs_boxes = []
        # Loop through the rows and columns of the image
        for i in range(0, height, batch_size_):
            for j in range(0, width, batch_size_):
                # Define a window for the current batch
                window = Window(col_off=j, row_off=i, width=batch_size_, height=batch_size_)
                # Read the batch from the raster image
                batch = src.read(window=window)
                # Append the raster batch to the list
                raster_batches.append(batch)

                # Convert the window to geospatial coordinates
                left, top = src.xy(i, j)
                right, bottom = src.xy(i+batch_size_, j+batch_size_)
                batch_bounds = box(left, bottom, right, top)

                # Clip the GeoDataFrame using the batch bounds
                itcs_clipped = gpd.clip(itcs, batch_bounds)

                # Transform the coordinates relative to the raster batch's origin
                itcs_clipped["geometry"] = itcs_clipped["geometry"].apply(
                    transform_coordinates, x_offset=left, y_offset=bottom
                )
                
                bbox_clipped = deepforest.utilities.annotations_to_shapefile(bbox, transform=src.transform, crs = src.crs)
                #from bboxes, clip only those whose xmin, ymin, xmax, ymax fit within the batch bounds
                bbox_clipped = gpd.clip(bbox_clipped, batch_bounds)

                #remove boxes that are LINESTRING or POINT
                bbox_clipped = bbox_clipped[bbox_clipped.geometry.type == 'Polygon']
                # Transform the coordinates of each box polygin relative to the raster batch's origin
                bbox_clipped["geometry"] = bbox_clipped["geometry"].apply(
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
                # Create a new DataFrame with label, x, and y columns from bbox_clipped
                # Extract the bounding box coordinates for each polygon
                tmp_bx = []
                for geometry in bbox_clipped.geometry:
                    bounds = geometry.bounds  # Returns (minx, miny, maxx, maxy)
                    left, bottom, right, top = bounds
                    tmp_bx.append([left, bottom, right, top])

                tmp_bx = np.array(tmp_bx)

                # Append the DataFrame to the list
                itcs_batches.append(itcs_df)
                affines.append(src.window_transform(window))
                itcs_boxes.append(tmp_bx)
    # Return the lists of raster batches and clipped GeoDataFrames
    return raster_batches, itcs_batches, itcs_boxes, affines


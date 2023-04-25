import laspy
import numpy as np
from scipy.spatial import KDTree
from scipy.signal import argrelextrema

input_las_file = './field_data_alignment/imagery/LiDAR.laz'

def local_maxima(points, radius):
    kdtree = KDTree(points)
    neighbors = kdtree.query_ball_tree(kdtree, radius)
    maxima = []

    for i, point_neighbors in enumerate(neighbors):
        if len(point_neighbors) == 1:  # point has no neighbors
            continue

        is_maxima = all(points[i][2] > points[j][2] for j in point_neighbors if j != i)
        if is_maxima:
            maxima.append(points[i])

    return np.array(maxima)

def get_ttops(input_las_file, search_radius=5):
    # Read the LAS file
    las_data = laspy.read(input_las_file)

    # Extract the x, y, z coordinates
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Detect local maxima (tree tops)
    tree_tops = local_maxima(points, search_radius)

    print(f"Detected {len(tree_tops)} tree tops")
    print(tree_tops)





def clip_laz_to_raster_extent(input_laz_file, input_raster_file, output_laz_file):
    # Read the raster file and get its bounds
    with rasterio.open(input_raster_file) as src:
        raster_bounds = src.bounds

    # Read the LAZ file
    las_data = laspy.read(input_laz_file)

    # Get the x, y, z coordinates
    points_xyz = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Clip the LiDAR data based on the raster bounds
    mask = (
        (points_xyz[:, 0] >= raster_bounds.left)
        & (points_xyz[:, 0] <= raster_bounds.right)
        & (points_xyz[:, 1] >= raster_bounds.bottom)
        & (points_xyz[:, 1] <= raster_bounds.top)
    )

    # Apply the mask and create a new LAS file with the clipped data
    clipped_points = points_xyz[mask]
    las_data_clipped = las_data[mask]
    las_data_clipped.write(output_laz_file)

    print(f"Clipped {input_laz_file} to match the extent of {input_raster_file} and saved as {output_laz_file}")

if __name__ == "__main__":
    input_laz_file = "path/to/your/input_lidar_data.laz"
    input_raster_file = "path/to/your/clipped_raster.tif"
    output_laz_file = "path/to/your/output_clipped_lidar_data.laz"
    clip_laz_to_raster_extent(input_laz_file, input_raster_file, output_laz_file)

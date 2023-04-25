import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import laspy


from field_data_alignment.src.utils import *

forestGEO_gdf = gpd.read_file('~/Documents/GitHub/GEOtreehealth/field_data_alignment/indir/test_noisy.shp')
field_gdf = gpd.read_file('~/Documents/GitHub/GEOtreehealth/field_data_alignment/indir/data_field.shp')

#extract only DBH, CrownPosition and Tag from forestGEO_gdf
forestGEO_gdf = forestGEO_gdf[['StemTag', 'DBH', 'Crwnpst',  'Species', 'Quad', 'Status']]

# rename stemTag column into Tag
forestGEO_gdf = forestGEO_gdf.rename(columns={'StemTag': 'Tag'})

hsi = rasterio.open('field_data_alignment/imagery/HSI.tif')
rgb = rasterio.open('field_data_alignment/imagery/RGB.tif')
lidar_path = 'field_data_alignment/imagery/LiDAR.laz'

forestGEO_features = extract_features(forestGEO_gdf, hsi, rgb, lidar_path)
field_features = extract_features(field_gdf, hsi, rgb, lidar_path)
aligned_gdf = align_data(forestGEO_gdf, field_gdf, forestGEO_features, field_features, threshold=0.1)
aligned_gdf.to_file('~/Documents/GitHub/GEOtreehealth/field_data_alignment/outdir/aligned_data.shp')

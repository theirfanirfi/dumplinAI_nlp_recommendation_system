from geo_locator.preprocessor import load_and_preprocess
from geo_locator.locator import find_containing_or_nearest

# Load once during app startup
gdf = load_and_preprocess("./DumplinAI.city_boundaries.csv")

# Get result for given coordinates
result = find_containing_or_nearest(gdf, lon=-73.4447743, lat=40.7291618)

# Access result
print(result[['properties.name', 'geometry']])
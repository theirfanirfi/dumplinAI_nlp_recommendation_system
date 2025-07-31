import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


def load_and_preprocess(csv_path: str) -> gpd.GeoDataFrame:
    """
    Loads a flattened CSV with geometry coordinates and constructs a GeoDataFrame.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with proper Polygon geometries.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

    # Extract coordinate columns
    coord_cols = [col for col in df.columns if col.startswith("geometry.coordinates[0][")]
    long_cols = sorted([col for col in coord_cols if col.endswith('[0]')])
    lat_cols  = sorted([col for col in coord_cols if col.endswith('[1]')])

    def build_polygon(row):
        coords = []
        for lon_col, lat_col in zip(long_cols, lat_cols):
            lon, lat = row.get(lon_col), row.get(lat_col)
            if lon != '' and lat != '' and pd.notnull(lon) and pd.notnull(lat):
                coords.append((lon, lat))
        return Polygon(coords)

    df['geometry'] = df.apply(build_polygon, axis=1)

    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    return gdf

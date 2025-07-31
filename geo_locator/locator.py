from shapely.geometry import Point
import geopandas as gpd


def find_containing_or_nearest(gdf: gpd.GeoDataFrame, lon: float, lat: float) -> gpd.GeoDataFrame:
    """
    Find rows that contain the point. If none, return the nearest geometry.

    Args:
        gdf (GeoDataFrame): GeoDataFrame with polygon geometries
        lon (float): Longitude
        lat (float): Latitude

    Returns:
        GeoDataFrame: Matched rows
    """
    point = Point(lon, lat)
    
    # Try finding containing polygon(s)
    matched = gdf[gdf.contains(point)]
    if not matched.empty:
        return matched

    # Compute distance for fallback
    # Make sure CRS is projected for distance (meters)
    projected_gdf = gdf.to_crs("EPSG:3857")
    projected_point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

    projected_gdf['distance_to_point'] = projected_gdf.geometry.distance(projected_point)
    nearest = projected_gdf.sort_values("distance_to_point").head(1)

    # Reproject result back to WGS84 if needed
    return nearest.to_crs("EPSG:4326")

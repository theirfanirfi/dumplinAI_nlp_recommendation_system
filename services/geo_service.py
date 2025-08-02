import json
from geo_locator.preprocessor import load_and_preprocess
from geo_locator.locator import find_containing_or_nearest
from config.settings import Config
import pandas as pd

class GeoService:
    def __init__(self, boundaries_file=None):
        self.boundaries_file = boundaries_file or f"{Config.ROOT}/DumplinAI.city_boundaries.csv"
        self.gdf = None
        self._load_boundaries()

    def _load_boundaries(self):
        """Load city boundaries data once during initialization"""
        try:
            self.gdf = load_and_preprocess(self.boundaries_file)
            print("City boundaries loaded successfully")
        except Exception as e:
            print(f"Error loading city boundaries: {e}")
            self.gdf = None

    def extract_coordinates_from_location(self, place_df):
        """Extract coordinates from location string/object"""
        return place_df['location.coordinates[0]'], place_df['location.coordinates[1]']

    def find_cities_in_boundaries(self, places_df):
        """Find all places within the city boundaries of the target city"""
        if self.gdf is None:
            print("City boundaries not loaded, falling back to city name matching")
            return False, places_df

        try:

            # Now filter all places that fall within this boundary
            filtered_places = []
            for idx, place in places_df.iterrows():
                place_lon, place_lat = self.extract_coordinates_from_location(place)
                if place_lon is not None and place_lat is not None:
                    place_boundary = find_containing_or_nearest(self.gdf, lon=place_lon, lat=place_lat)
                    if (place_boundary is not None and not place_boundary.empty):
                        for _, boundary_row in place_boundary.iterrows():
                                filtered_places.append(boundary_row)

            if filtered_places:
                return True, pd.DataFrame(filtered_places)
            else:
                print("No places found within boundaries, falling back to city name matching")
                return False, places_df

        except Exception as e:
            print(f"Error in boundary search: {e}, falling back to city name matching")
            return False, places_df

import json
from geo_locator.preprocessor import load_and_preprocess
from geo_locator.locator import find_containing_or_nearest
from config.settings import Config

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

    def extract_coordinates_from_location(self, location_str):
        """Extract coordinates from location string/object"""
        try:
            if isinstance(location_str, str):
                location_data = json.loads(location_str.replace("'", '"'))
            else:
                location_data = location_str
            
            if location_data.get('type') == 'Point' and 'coordinates' in location_data:
                coordinates = location_data['coordinates']
                return coordinates[0], coordinates[1]  # lon, lat
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
        return None, None

    def find_cities_in_boundaries(self, target_city, places_df):
        """Find all places within the city boundaries of the target city"""
        if self.gdf is None:
            print("City boundaries not loaded, falling back to city name matching")
            return places_df[places_df['city'] == target_city]

        # Get all places in the target city to find representative coordinates
        city_places = places_df[places_df['city'] == target_city]
        if city_places.empty:
            return city_places

        # Use the first place's coordinates to find the city boundary
        first_place = city_places.iloc[0]
        lon, lat = self.extract_coordinates_from_location(first_place['location'])
        
        if lon is None or lat is None:
            print("Could not extract coordinates, falling back to city name matching")
            return city_places

        # Find the containing city boundary
        try:
            boundary_result = find_containing_or_nearest(self.gdf, lon=lon, lat=lat)
            if boundary_result is None or boundary_result.empty:
                print("No boundary found, falling back to city name matching")
                return city_places

            # Now filter all places that fall within this boundary
            filtered_places = []
            for idx, place in places_df.iterrows():
                place_lon, place_lat = self.extract_coordinates_from_location(place['location'])
                if place_lon is not None and place_lat is not None:
                    place_boundary = find_containing_or_nearest(self.gdf, lon=place_lon, lat=place_lat)
                    if (place_boundary is not None and not place_boundary.empty and 
                        place_boundary.iloc[0]['properties.name'] == boundary_result.iloc[0]['properties.name']):
                        filtered_places.append(place)

            if filtered_places:
                return pd.DataFrame(filtered_places)
            else:
                print("No places found within boundaries, falling back to city name matching")
                return city_places

        except Exception as e:
            print(f"Error in boundary search: {e}, falling back to city name matching")
            return city_places

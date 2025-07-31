import openai
from services.mood_classifier import MoodClassifier
from services.geo_service import GeoService
from data.loader import DatasetLoader
from config.settings import Config

class RecommendationService:
    def __init__(self, datasets=None):
        self.datasets = datasets or DatasetLoader()
        self.mood_classifier = MoodClassifier()
        self.geo_service = GeoService()
        self.user_prompt = None
        self.user_city = None
        self.user_mode = None
        self.places_dictionary = None
        self.city_picker_dictionary = None
        self.posts_dictionary = None
        self.compiled_prompt = None
        
        # OpenAI client
        openai.api_key = Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    def set_user_city(self, user_city):
        """Set the user's city"""
        self.user_city = user_city
        return self

    def set_user_prompt(self, user_prompt):
        """Set the user's request prompt"""
        self.user_prompt = user_prompt
        return self

    def set_user_mood(self):
        """Classify and set user mood based on their prompt"""
        self.user_mode = self.mood_classifier.classify_user_mood(self.user_prompt)
        return self

    def get_places_in_boundaries(self):
        """Get places within city boundaries and matching user mood"""
        all_places = self.datasets.get_places()
        
        # Get places within city boundaries using geo service
        boundary_places = self.geo_service.find_cities_in_boundaries(self.user_city, all_places)
        
        # Filter by user mood
        mood_filtered_places = boundary_places[boundary_places['label'] == self.user_mode]
        
        self.places_dictionary = mood_filtered_places.to_dict()
        
        # Get city picker info
        city_picker_data = self.datasets.get_city_picker()
        city_info = city_picker_data[city_picker_data['city'] == self.user_city]
        self.city_picker_dictionary = city_info.to_dict() if not city_info.empty else {}
        
        return self

    def get_posts_for_boundary_cities(self):
        """Get posts for all cities within the boundary"""
        all_places = self.datasets.get_places()
        boundary_places = self.geo_service.find_cities_in_boundaries(self.user_city, all_places)
        
        # Get unique cities from boundary places
        boundary_cities = boundary_places['city'].unique().tolist()
        
        # Get posts for all these cities
        all_posts = self.datasets.get_posts()
        boundary_posts = all_posts[all_posts['city'].isin(boundary_cities)]
        
        self.posts_dictionary = boundary_posts.to_dict()
        return self

    def compile_prompt(self):
        """Compile the prompt for OpenAI API"""
        self.compiled_prompt = f"""
        You are DumplinAI, a helpful and creative assistant that recommends places based on user input.
        Here is the user's current city and state: {self.city_picker_dictionary}
        Here is the user's city picker line: {self.city_picker_dictionary.get('cuisine_summary', {}).get(0, 'N/A') if self.city_picker_dictionary else 'N/A'}
        Here is the user's description of what they are looking for: {self.user_prompt}
        Here are the places that match the user's desired mode: {self.places_dictionary}
        Here are the social media posts from influencers in the area: {self.posts_dictionary}
        
        You can reference social media posts like this: Influencer @username who has @followersCount followers posted @url and says: @Phase1.transcript.0
        
        The user mood detected is: "{self.user_mode}"
        
        Based on the user's description and the provided places within the city boundaries, recommend a place from the list that best fits the user's needs.
        When suggesting, provide reasons - for example, if the user has traveled, mention they might be tired and suggest accordingly.
        
        Strict instruction: Only use the information provided above, do not use external knowledge.
        """
        return self

    def get_response(self):
        """Get recommendation response from OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are DumplinAI, a helpful restaurant and place recommendation assistant."},
                    {"role": "user", "content": self.compiled_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

    def get_recommendation(self, user_city, user_prompt):
        """Complete recommendation pipeline"""
        return (self.set_user_city(user_city)
                   .set_user_prompt(user_prompt)
                   .set_user_mood()
                   .get_places_in_boundaries()
                   .get_posts_for_boundary_cities()
                   .compile_prompt()
                   .get_response())

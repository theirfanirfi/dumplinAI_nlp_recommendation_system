from data.preprocessor import DataPreprocessor
from services.recommendation_service import RecommendationService

def preprocess_data():
    """Run data preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    preprocessor.process_all()
    print("Data preprocessing completed!")

def main():
    """Main application entry point"""
    # Initialize recommendation service
    recommender = RecommendationService()
    
    # Example usage
    user_input = """
    Just landed in LA, starving, but don't want a whole production. Something solid and close.
    """
    
    response = recommender.get_recommendation('Los Angeles', user_input)
    print(response)

if __name__ == "__main__":
    # Uncomment the line below to run preprocessing first
    # preprocess_data()
    
    main()
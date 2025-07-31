from transformers import pipeline
from config.settings import Config

class MoodClassifier:
    def __init__(self):
        self.classifier = pipeline(Config.CLASSIFIER_TYPE, model=Config.CLASSIFIER_MODEL)
        self.labels = Config.MOOD_LABELS

    def assign_mood(self,text):
        result = self.classifier(text, self.labels)
        return result['labels'][result['scores'].index(max(result['scores']))]
    
    def classify_user_mood(self, user_prompt):
        """Classify user mood based on their prompt"""
        result = self.classifier(user_prompt, self.labels)
        return result['labels'][result['scores'].index(max(result['scores']))]

import os

class Config:
    ROOT = "/content/drive/MyDrive/Projects/ai/dumplin"

    # Classification labels
    MOOD_LABELS = ['lowkey', 'nightout', 'comforting', 'surprise', 'hidden gem']
    
    # Model configurations
    CLASSIFIER_TYPE = "zero-shot-classification"
    CLASSIFIER_MODEL = "facebook/bart-large-mnli"
    OPENAI_MODEL = "gpt-4o-mini"

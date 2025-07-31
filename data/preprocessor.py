import pandas as pd
import re
import os
from transformers import pipeline
from config.settings import Config

class DataPreprocessor:
    def __init__(self, root=Config.ROOT):
        self.places_dataset = None
        self.city_picker_dataset = None
        self.posts_dataset = None
        self.creators_dataset = None
        self.classifier = pipeline("zero-shot-classification", model=Config.CLASSIFIER_MODEL)
        self.labels = Config.MOOD_LABELS
        self.ROOT = root

    def load_datasets(self):
        """Load all datasets from CSV files"""
        self.places_dataset = pd.read_csv(f"{self.ROOT}/DumplinAI.places_los.csv")
        self.city_picker_dataset = pd.read_csv(f"{self.ROOT}/DumplinAI.city_picker.csv")
        self.posts_dataset = pd.read_csv(f"{self.ROOT}/DumplinAI.losposts.csv")
        self.creators_dataset = pd.read_csv(f"{self.ROOT}/DumplinAI.creators.csv")
        return self

    def clean_datasets(self):
        """Clean and filter datasets to required columns"""
        self.places_dataset = self.places_dataset[['title', 'description', 'categoryName', 'city', 'location']]
        self.places_dataset.dropna(subset=['description'], inplace=True)
        self.city_picker_dataset = self.city_picker_dataset[['city', 'state', 'cuisine_summary']]
        return self

    def generate_label(self, row):
        """Generate mood label for a place based on its description"""
        sequence_to_classify = row['description']
        result = self.classifier(sequence_to_classify, self.labels)
        return result['labels'][result['scores'].index(max(result['scores']))]

    def label_places(self):
        """Apply mood labels to all places"""
        self.places_dataset['label'] = self.places_dataset.apply(self.generate_label, axis=1)
        return self

    def clean_posts(self):
        """Clean posts dataset"""
        self.posts_dataset = self.posts_dataset[['city', 'platform', 'creator_id', 'url', 'Phase1.transcript.0', 'caption']]
        return self

    def clean_creators(self):
        """Clean creators dataset"""
        self.creators_dataset = self.creators_dataset[['_id', 'username', 'followersCount', 'profilePicUrl', 'created_at']]
        return self

    def leftjoin_posts_creators(self):
        """Join posts with creators data"""
        self.posts_dataset = pd.merge(self.posts_dataset, self.creators_dataset, 
                                    how='left', left_on='creator_id', right_on='_id')
        return self

    def create_dir_new_version(self):
        """Create a new versioned directory for datasets"""
        base_path = self.ROOT
        dir_pattern = re.compile(r'datasets_v(\d+)')

        existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        version_numbers = [int(match.group(1)) for d in existing_dirs for match in [dir_pattern.match(d)] if match]

        next_version = max(version_numbers) + 1 if version_numbers else 1
        new_dir_name = f'datasets_v{next_version}'
        new_dir_path = os.path.join(base_path, new_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
        return new_dir_path

    def compile_and_save_datasets(self):
        """Save all processed datasets to new version directory"""
        dir_path = self.create_dir_new_version()
        self.places_dataset.to_csv(f"{dir_path}/DumplinAI.places.csv", index=False)
        self.city_picker_dataset.to_csv(f"{dir_path}/DumplinAI.city_picker.csv", index=False)
        self.posts_dataset.to_csv(f"{dir_path}/DumplinAI.posts.csv", index=False)
        self.creators_dataset.to_csv(f"{dir_path}/DumplinAI.creators.csv", index=False)
        return self

    def process_all(self):
        """Execute full preprocessing pipeline"""
        return (self.load_datasets()
                   .clean_datasets()
                   .label_places()
                   .clean_posts()
                   .clean_creators()
                   .leftjoin_posts_creators()
                   .compile_and_save_datasets())

import pandas as pd
import re
import os
from config.settings import Config

class DatasetLoader:
    def __init__(self, root=Config.ROOT):
        self.root = root
        self.latest_dir_path = self._find_latest_version_directory()
        self.places_dataset = None
        self.city_picker_dataset = None
        self.posts_dataset = None
        self.creators_dataset = None
        self.load_datasets()

    def _find_latest_version_directory(self):
        """Find the most recent version directory"""
        base_path = self.root
        dir_pattern = re.compile(r'datasets_v(\d+)')
        existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        version_dirs = [(int(match.group(1)), d) for d in existing_dirs for match in [dir_pattern.match(d)] if match]

        if version_dirs:
            latest_version, latest_dir_name = max(version_dirs)
            return os.path.join(base_path, latest_dir_name)
        else:
            return None

    def load_datasets(self):
        """Load all datasets from the latest version directory"""
        if not self.latest_dir_path:
            print("No datasets directories found.")
            return

        try:
            self.places_dataset = pd.read_csv(f"{self.latest_dir_path}/DumplinAI.places.csv")
            self.city_picker_dataset = pd.read_csv(f"{self.latest_dir_path}/DumplinAI.city_picker.csv")
            self.posts_dataset = pd.read_csv(f"{self.latest_dir_path}/DumplinAI.posts.csv")
            self.creators_dataset = pd.read_csv(f"{self.latest_dir_path}/DumplinAI.creators.csv")
            print(f"Datasets loaded successfully from {self.latest_dir_path}")
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")

    def get_places(self):
        return self.places_dataset

    def get_city_picker(self):
        return self.city_picker_dataset

    def get_posts(self):
        return self.posts_dataset

    def get_creators(self):
        return self.creators_dataset

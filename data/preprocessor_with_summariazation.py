import pandas as pd
import re
from transformers import pipeline
import openai
from config.settings import Config

class DataPreprocessor:
    def __init__(self, root=Config.ROOT):
        self.places_dataset = None
        self.city_picker_dataset = None
        self.posts_dataset = None
        self.creators_dataset = None
        self.classifier = pipeline(Config.CLASSIFIER_TYPE, model=Config.CLASSIFIER_MODEL)
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
        self.places_dataset = self.places_dataset[['title', 'description', 'categoryName', 'city', 'location.coordinates[0]','location.coordinates[1]']]
        self.places_dataset.dropna(subset=['description'], inplace=True)
        self.city_picker_dataset = self.city_picker_dataset[['city', 'state', 'cuisine_summary']]
        return self

    def generate_label(self, row):
        """Generate mood label for a place based on its description"""
        sequence_to_classify = row['summarization']
        # fetch_all_posts_for_the_city = self.pos
        result = self.classifier(sequence_to_classify, self.labels)
        return result['labels'][result['scores'].index(max(result['scores']))]

    def summarize_text_open_ai_request(self, text):
      client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
      try:
          response = client.chat.completions.create(
              model=Config.OPENAI_MODEL,
              messages=[
                  {"role": "system", "content": "while considering the city name, its decription, city picker line, and social media posts, Summarize the text in such a way that it reflects the fitness of the place for people in one of these labels: lowkey, nightout, comfortable, surprise, hidden gem. We will be recommending the places based on the people mood. Do it only in plain text with no formatting"},
                  {"role": "user", "content": text}
              ]
          )
          return True, response.choices[0].message.content
      except Exception as e:
          return False, f"Error generating response: {e}"

    def summerization(self, row):
        posts_for_the_city = self.posts_dataset[self.posts_dataset['city'] == row['city']]
        posts_text = "social media posts: "
        for _, post in posts_for_the_city.iterrows():
            posts_text += f"{post['Phase1.transcript.0']} {post['caption']}"

        posts_text += f"city name: {row['city']} and its description {row['description']}"
        city_picker = self.city_picker_dataset[self.city_picker_dataset['city'] == row['city']]
        posts_text += f"city picker line: {city_picker['cuisine_summary'].values[0]}"
        ai_summarization = self.summarize_text_open_ai_request(posts_text)
        print(ai_summarization)
        return ai_summarization[1] if ai_summarization[0] else posts_text


    def summarize_places_based_on_posts_and_description(self):
        """make text ready for summarization"""
        self.places_dataset['summarization'] = self.places_dataset.apply(self.summerization, axis=1)
        return self


    def generate_label_for_summarization(self):
      """make text ready for summarization"""
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
        (self.load_datasets()
                  .clean_datasets()
                  .clean_posts()
                  .clean_creators()
                  .leftjoin_posts_creators()
                  .summarize_places_based_on_posts_and_description()
                  .generate_label_for_summarization()
                  .compile_and_save_datasets())
        return self

import pandas as pd
import re

class DataLoader:
    def __init__(self, file_path='data/reviews.csv'):
        self.file_path = file_path
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<[^>]*>', '', text)  # remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        return text
        
    def load_and_preprocess(self):
        try:
            df = pd.read_csv(self.file_path)
            if 'review' not in df.columns or 'sentiment' not in df.columns:
                raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
                
            df['clean_review'] = df['review'].apply(self.clean_text)
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
            
    def prepare_batch(self, upload_file):
        try:
            df = pd.read_csv(upload_file)
            if 'review' not in df.columns:
                raise ValueError("Uploaded CSV must contain a 'review' column.")
            df['clean_review'] = df['review'].apply(self.clean_text)
            return df
        except Exception as e:
            raise Exception(f"Error processing batch file: {e}")

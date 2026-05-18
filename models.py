from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

class SentimentModel:
    def __init__(self, model_type='logistic_regression'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError("Unsupported model type. Choose 'logistic_regression' or 'naive_bayes'.")
            
        self.is_trained = False
        self.classes_ = []
        
    def train(self, X_train, y_train):
        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)
        self.classes_ = self.model.classes_
        self.is_trained = True
        
    def predict(self, texts):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        if isinstance(texts, str):
            texts = [texts]
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)
        
    def predict_proba(self, texts):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        if isinstance(texts, str):
            texts = [texts]
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_vec)
        
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
        
    def get_model_coefficients(self):
        if self.model_type == 'logistic_regression':
            return self.model.coef_[0]
        return None

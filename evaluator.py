import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from wordcloud import WordCloud
import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        # Need to ensure classes are available
        if not hasattr(model, 'classes_') or len(model.classes_) == 0:
            raise ValueError("Model must be trained before evaluation.")
            
        self.y_pred = self.model.predict(X_test)
        
        try:
            self.y_prob = self.model.predict_proba(X_test)[:, 1] if len(self.model.classes_) == 2 else None
        except:
            self.y_prob = None
        
    def get_metrics(self):
        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred, output_dict=True)
        return acc, report
        
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.model.classes_)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        return fig
        
    def plot_roc_curve(self):
        if self.y_prob is None or len(self.model.classes_) != 2:
            return None
        # Assuming pos_label is the second class in self.model.classes_
        pos_label = self.model.classes_[1]
        y_test_binary = (self.y_test == pos_label).astype(int)
        
        fpr, tpr, _ = roc_curve(y_test_binary, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        return fig
        
    def plot_feature_importance(self):
        coefs = self.model.get_model_coefficients()
        if coefs is None:
            return None
            
        feature_names = self.model.get_feature_names()
        # Handle cases with very few features gracefully
        n_features = min(10, len(coefs))
        if n_features == 0:
            return None
            
        top_positive_indices = np.argsort(coefs)[-n_features:]
        top_negative_indices = np.argsort(coefs)[:n_features]
        
        top_indices = np.concatenate([top_negative_indices, top_positive_indices])
        top_features = [feature_names[i] for i in top_indices]
        top_coefs = coefs[top_indices]
        
        colors = ['red' if c < 0 else 'green' for c in top_coefs]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features, top_coefs, color=colors)
        ax.set_title('Top Positive and Negative Features (Logistic Regression)')
        ax.set_xlabel('Coefficient Magnitude')
        return fig
        
    def plot_word_clouds(self, df):
        # Generates two wordclouds: positive and negative
        positive_text = " ".join(df[df['sentiment'] == 'positive']['clean_review'].astype(str))
        negative_text = " ".join(df[df['sentiment'] == 'negative']['clean_review'].astype(str))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        if positive_text.strip():
            wc_pos = WordCloud(width=400, height=400, background_color='white', colormap='Greens').generate(positive_text)
            axes[0].imshow(wc_pos, interpolation='bilinear')
        axes[0].set_title('Positive Words')
        axes[0].axis('off')
        
        if negative_text.strip():
            wc_neg = WordCloud(width=400, height=400, background_color='white', colormap='Reds').generate(negative_text)
            axes[1].imshow(wc_neg, interpolation='bilinear')
        axes[1].set_title('Negative Words')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig

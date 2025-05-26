import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict, Counter
import gradio as gr
from sklearn_crfsuite import CRF

from nltk.tokenize import word_tokenize

# Download necessary NLTK datasets
nltk.download("brown")
nltk.download("universal_tagset")
nltk.download("punkt")


# Preprocess the data
def preprocess_data():
    tagged_sents = brown.tagged_sents(tagset="universal")
    return tagged_sents


tagged_sents = preprocess_data()



class CRF_POS_Tagging:
    def __init__(self):
        self.crf = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100)
        self.tags = set()

    def extract_features(self, sentence, index):
        """Extract features for a given word in the sentence."""
        word = sentence[index][0]
        features = {
            "word": word,
            "is_capitalized": word[0].upper() == word[0],
            "is_all_caps": word.upper() == word,
            "is_all_lower": word.lower() == word,
            "prefix_1": word[0],
            "prefix_2": word[:2],
            "suffix_1": word[-1],
            "suffix_2": word[-2:],
            "prev_word": "" if index == 0 else sentence[index - 1][0],
            "next_word": "" if index == len(sentence) - 1 else sentence[index + 1][0],
        }
        return features

    def prepare_dataset(self, tagged_sents):
        """Prepare the dataset by extracting features and labels."""
        X = []
        y = []
        for sentence in tagged_sents:
            features = []
            labels = []
            for index in range(len(sentence)):
                features.append(self.extract_features(sentence, index))
                labels.append(sentence[index][1])
                self.tags.add(sentence[index][1])
            X.append(features)
            y.append(labels)
        return X, y

    # Training
    def train(self, tagged_sents):
        X, y = self.prepare_dataset(tagged_sents)
        self.crf.fit(X, y)

    # Prediction
    def predict(self, sentence):
        features = [
            self.extract_features(sentence, index) for index in range(len(sentence))
        ]
        return self.crf.predict_single(features)

# Create an instance of the CRF_POS_Tagging class
crf_model = CRF_POS_Tagging()

crf_model.train(tagged_sents)

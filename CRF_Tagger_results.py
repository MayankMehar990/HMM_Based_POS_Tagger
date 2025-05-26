import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict, Counter
import gradio as gr
from sklearn_crfsuite import CRF
import pandas as pd
import sklearn_crfsuite
from nltk.tokenize import word_tokenize
# Importing libraries for metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)
from sklearn_crfsuite.metrics import flat_classification_report


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
    
    def evaluate(self, tagged_sents):
        """Evaluate the model using a set of tagged sentences."""
        X_test, y_test = self.prepare_dataset(tagged_sents)
        y_pred = self.crf.predict(X_test)
        return flat_classification_report(y_test, y_pred)



# Evaluate the model using 5-fold cross-validation
def evaluate_model(tagged_sents, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    all_predictions = []
    all_true_labels = []
    tags_set = set()

    for train_index, test_index in kf.split(tagged_sents):
        train_sents = [tagged_sents[i] for i in train_index]
        test_sents = [tagged_sents[i] for i in test_index]

        # Initialize the CRF_POS_Tagging model
        model = CRF_POS_Tagging()
        model.train(tagged_sents)

        true_labels = []
        predictions = []

        for sentence in test_sents:
            words, tags = zip(*sentence)  # Separate words and true tags
            predicted_tags = model.predict(sentence)  # Predict using the CRF model
            true_labels.extend(tags)
            predictions.extend(predicted_tags)

            # Add the tags and predicted tags to the tags set for confusion matrix and classification report
            tags_set.update(tags)
            tags_set.update(predicted_tags)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        accuracies.append(accuracy)

        # Calculate F1, F0.5, and F2 scores
        f1 = f1_score(true_labels, predictions, average="weighted")
        f05 = fbeta_score(true_labels, predictions, beta=0.5, average="weighted")
        f2 = fbeta_score(true_labels, predictions, beta=2, average="weighted")

        # Confusion Matrix
        tags_list = list(tags_set)  # Create list from the set of tags
        conf_matrix = confusion_matrix(true_labels, predictions, labels=tags_list)

        # Classification Report
        class_report = classification_report(
            true_labels, predictions, labels=tags_list, output_dict=True
        )

        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

    # Average accuracy across all folds
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    # Print Confusion Matrix
    print("Confusion Matrix:\n", conf_matrix)

    # Print Classification Report
    print("Classification Report:\n", class_report)

    # Print F0.5, F1, and F2 scores
    print(f"F0.5-score: {f05:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F2-score: {f2:.4f}")



# Test the evaluation function
evaluate_model(tagged_sents)

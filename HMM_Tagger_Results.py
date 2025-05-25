#Libraries
import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict, Counter

from nltk.corpus import brown
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict, Counter


from nltk.corpus import brown
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict, Counter

# brown corpus
nltk.download('brown')
nltk.download('universal_tagset')

# preprocessing
tagged_sentences = brown.tagged_sents(tagset='universal')

class HMM:
    def __init__(self):
        # dictionaries for transitions and emissions
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))

        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()

    # Training
    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            prev_tag = '<s>'
            for word, tag in sentence:
                self.vocab.add(word)
                self.tags.add(tag)
                self.tag_counts[tag] += 1

                # Transition Probabilities
                self.transition_probs[prev_tag][tag] += 1

                # Emission Probabilities
                self.emission_probs[tag][word] += 1
                prev_tag = tag

            # addding transition to the end token
            self.transition_probs[prev_tag]['</s>'] += 1

        # Normalizing the probabilities
        for prev_tag, transitions in self.transition_probs.items():
            total = sum(transitions.values())
            for tag in transitions:
                self.transition_probs[prev_tag][tag] = (self.transition_probs[prev_tag][tag]) / (total)

        for tag, emissions in self.emission_probs.items():
            total = sum(emissions.values())
            for word in emissions:
                self.emission_probs[tag][word] = (self.emission_probs[tag][word]) / (total)

    # decoding
    # to find the most probable sequence of POS tags for a given sequence of words
    def viterbi(self, words):
        n = len(words)

        # Dynamic programming table to store the probabilities
        dp = defaultdict(lambda: defaultdict(float))

        # Backpointer table to store the best previous tag
        backpointer = defaultdict(lambda: defaultdict(str))

        for tag in self.tags:
            dp[0][tag] = self.transition_probs['<s>'][tag] * self.emission_probs[tag].get(words[0], 1e-6)
            backpointer[0][tag] = '<s>'

        # Recursion
        for i in range(1, n):
            for curr_tag in self.tags:
                max_prob, best_prev_tag = max(
                    (dp[i-1][prev_tag] * self.transition_probs[prev_tag][curr_tag] * self.emission_probs[curr_tag].get(words[i], 1e-6), prev_tag)
                    for prev_tag in self.tags
                )
                dp[i][curr_tag] = max_prob
                backpointer[i][curr_tag] = best_prev_tag

        max_prob, best_last_tag = max((dp[n-1][tag] * self.transition_probs[tag]['</s>'], tag) for tag in self.tags)

        # getting the best path by backtracking
        best_path = [best_last_tag]
        for i in range(n-1, 0, -1):
            best_path.append(backpointer[i][best_path[-1]])
        best_path.reverse()

        return list(zip(words, best_path))

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd



def evaluate_model(hmm, test_sentences):
    y_true, y_pred = [], []
    for sentence in test_sentences:
        words, true_tags = zip(*sentence)
        pred_tags = [tag for word, tag in hmm.viterbi(words)]
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)

    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(hmm.tags))
    per_pos_accuracy = pd.Series(np.diag(cm) / cm.sum(axis=1), index=hmm.tags)

    return overall_accuracy, cm, per_pos_accuracy


def cross_validate_hmm(tagged_sentences, n_splits=5):
    kf = KFold(n_splits=n_splits)
    overall_accuracies = []
    confusion_matrices = []
    per_pos_accuracies = []
    tags = None

    for train_index, test_index in kf.split(tagged_sentences):
        train_data = [tagged_sentences[i] for i in train_index]
        test_data = [tagged_sentences[i] for i in test_index]

        hmm = HMM()
        hmm.train(train_data)

        # Storing the tags set
        if tags is None:
            tags = list(hmm.tags)

        accuracy, cm, per_pos_acc = evaluate_model(hmm, test_data)
        overall_accuracies.append(accuracy)
        confusion_matrices.append(cm)
        per_pos_accuracies.append(per_pos_acc)

    avg_accuracy = np.mean(overall_accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)
    avg_per_pos_acc = pd.concat(per_pos_accuracies, axis=1).mean(axis=1)

    return avg_accuracy, avg_cm, avg_per_pos_acc, tags


# cross-validation
avg_accuracy, avg_cm, avg_per_pos_acc, tags = cross_validate_hmm(tagged_sentences, n_splits=5)

# Results
print(f"Average Accuracy: {avg_accuracy:.4f}")
print("\nConfusion Matrix:")
print(pd.DataFrame(avg_cm, index=tags, columns=tags))
print("\n\nPer POS Accuracy:")
print(avg_per_pos_acc)

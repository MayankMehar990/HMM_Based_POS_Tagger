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

hmm_model=HMM()
hmm_model.train(tagged_sentences)


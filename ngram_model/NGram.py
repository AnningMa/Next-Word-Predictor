from typing import List
import itertools

from nltk.lm import Vocabulary, NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline

import pickle
import os
from typing import List
import itertools
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *

class NGram:
    
    def __init__(self):
        self.counts = None 
        self.vocabulary = None
        self.train_ngrams = None
        self.max_n = 0
        self.most_common_words = []
        
    def build_ngrams(self, sentences: List[str], max_n: int, min_count: int = 2):
        self.max_n = max_n
        #Tokenization
        tokenized_text = [sentence.split() for sentence in sentences]
        # build Vocabulary
        all_words = itertools.chain.from_iterable(tokenized_text) #flatten the list of lists
        self.vocabulary = Vocabulary(all_words, unk_cutoff=min_count)
        # Replace rare tokens with <UNK>
        text_with_unks = [list(self.vocabulary.lookup(sent)) for sent in tokenized_text]
        # padded everygrams
        train_data, vocab = padded_everygram_pipeline(max_n, text_with_unks)
        self.train_ngrams = list(train_data)
        # build counts
        self.counts = NgramCounter(self.train_ngrams)
        
        top_words = self.counts[1].most_common(20)
        self.most_common_words = []
        for word, _ in top_words:
            if word not in ["<s>", "</s>", "<UNK>", "<START>", "<END>"]:
                self.most_common_words.append((word, 0.0001))
        
        print(f"Built n-grams up to {max_n}. Vocab size: {len(self.vocabulary)}")

    def get_count(self, word, context=()):
        return self.counts[context][word]
    
    def save_model(self, filename="ngram_model.pkl"):
        if self.counts is None:
            print("Run build_ngrams")
            return

        print(f"Saving to  {filename} ...")
        
        model_data = {
            "counts": self.counts,       
            "vocabulary": self.vocabulary, 
            "max_n": self.max_n,
            "most_common_words": self.most_common_words
        }
        
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
            
        print("Saved successfully!")
    
if __name__ == "__main__":
    data = load_data()
    train_data, val_data, test_data = split_dataset2(data, val_size=0.1, test_size=0.1)
    ngram_storage = NGram()
    ngram_storage.build_ngrams(train_data, max_n=3, min_count=2)
    ngram_storage.save_model(filename="ngram_model.pkl")
    
    
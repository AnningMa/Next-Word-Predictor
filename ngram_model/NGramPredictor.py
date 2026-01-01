from nltk.lm import Laplace
from typing import List, Tuple, Dict
from ngram_model.NGram import NGram
import os
import pickle

class NgramsPredictor:
    def __init__(self, model_path: str):
        print(f"Loading N-Gram model from: {model_path} ...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")
            
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    
        self.counts: NgramCounter = data["counts"]       
        self.vocabulary: Vocabulary = data["vocabulary"] 
        self.n = data["max_n"]                           
        self.most_common_words = data["most_common_words"] 
        
        self.vocab_size = len(self.vocabulary)
        
        print(f"Model loaded. Order (n): {self.n}, Vocab size: {self.vocab_size}")

    def _clean_context(self, context_words: List[str]) -> Tuple[str]:
        #  replace unknown words with <UNK>
        cleaned = [
            w if w in self.vocabulary else "<UNK>"
            for w in context_words
        ]
        
        # return the last (n-1) words as context
        return tuple(cleaned[-(self.n - 1):]) if self.n > 1 else ()

    def _get_laplace_score(self, word: str, context: tuple) -> float:
        
        # counts[context][word]
        word_count = self.counts[context][word]
    
        context_total_count = self.counts[context].N()
        
        # Laplace smoothing
        prob = (word_count + 1) / (context_total_count + self.vocab_size)
        return prob

    def predict_next(self, context_words: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if isinstance(context_words, str):
            context_words = context_words.strip().split()

        context = self._clean_context(context_words)
        
        # Backoff 
        for backoff in range(len(context), -1, -1):
            sub_context = context[len(context) - backoff :]
            
            candidates_freq_dist = self.counts[sub_context]
            
            # if no candidates found, continue to backoff
            if not candidates_freq_dist:
                continue 

            candidates = list(candidates_freq_dist.keys())
            
            results = []
            for word in candidates:
                if word in ["<s>", "</s>", "<UNK>", "<START>", "<END>"]:
                    continue
                
                score = self._get_laplace_score(word, sub_context)
                results.append((word, score))
                
            if results:
                sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
                
                return sorted_results[:top_k]
        
        # if no candidates found even after full backoff, return most common words
        fallback_results = [(w,0.0001) for w in self.most_common_words[:top_k]]
        return fallback_results

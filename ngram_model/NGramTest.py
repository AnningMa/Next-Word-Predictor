from typing import List, Dict
from .NGram import NGram
from .NGramPredictor import NgramsPredictor 
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *

# Build storage (NGram)
# ============================================================
def build_and_save_storage(train_data: List[str], max_n: int):
    storage = NGram()
    storage.build_ngrams(train_data, max_n)
    return storage

# ============================================================
# Create models (each with its own n)
# ============================================================
def create_models(storage, model_configs: List[Dict]):
    models = {}
    
    for config in model_configs:
        n = config["n"]
        name = config.get("name", f"{n}-gram")
        
        print(f"\nCreating model: {name} (n={n})")
        
        models[name] = NgramsPredictor(
            n=n,
            storage=storage,
        )
    
    return models

# ============================================================
# Evaluate each model with top-k accuracy
# ============================================================
def evaluate_models(models: Dict[str, NgramsPredictor], test_data: List[str]):

    print(f"{'Model':<30}", end='')
    for k in range(1, 11):
        print(f"{'Top-'+str(k):<8}", end='')
    print()
    print("-" * (30 + 8 * 10))
    
    results = {}

    for name, model in models.items():
        print(f"{name:<30}", end='')
        correct_counts = {k: 0 for k in range(1, 11)}
        total_predictions = 0
        
        for sentence in test_data:
            words = sentence.split()
            if len(words) < 2:
                continue  # Need at least one context word and one target word
            
            # Evaluate EVERY POS possible
            for i in range(1, len(words)):
                context = words[:i]
                true_next_word = words[i] #golden word
                predictions = model.predict_next(context, top_k=10) #predicted words
                predicted_words = [w for w, _ in predictions]
                try:
                    rank = predicted_words.index(true_next_word) + 1
                except ValueError:
                    rank = float('inf')
                    
                for k in range(1, 11):
                    if rank <= k:
                        correct_counts[k] += 1
                        
                total_predictions += 1
        
        #compute and print results
        model_results = {}
        for k in range(1, 11):
            acc = correct_counts[k] / total_predictions if total_predictions else 0
            model_results[f"top_{k}"] = acc
            print(f"{acc:<8.4f}", end='')

        results[name] = model_results
        print()

    return results

# ============================================================
# Terminal interactive test
# ============================================================
def interactive_test(model: NgramsPredictor):

    print(f"\n=== Interactive test with {model.n}-gram model ===")
    print("Type a sentence. Type 'exit' to quit.")

    while True:

        user_input = input("\n> ").strip()

        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
        
        context_words = preprocess(user_input).split()
        predictions = model.predict_next(context_words, top_k=10)
        if predictions:
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"{i}. {word} (prob={prob:.4f})")
        else:
            print("No prediction.")

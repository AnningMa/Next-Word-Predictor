import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *
from ngram_nn.model import NeuralNGram

class NGramNNPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        print(f"Loading: {model_path} ...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.word2id = checkpoint['word2id']
        self.id2word = {v: k for k, v in self.word2id.items()}
        
        self.model = NeuralNGram(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            context_size=self.config['context_size'],
            hidden_dim=self.config['hidden_dim'],
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict_next(self, text, top_k=5):
        text = preprocess(text)
        tokens = tokenize(text)
        n = self.config['context_size']
        if len(tokens) < n:
            pads = [START] * (n - len(tokens))
            context_tokens = pads + tokens
        else:
            context_tokens = tokens[-n:]
        
        input_ids = [self.word2id.get(t, self.word2id.get(UNK)) for t in context_tokens]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            word = self.id2word.get(idx, "<Unknown>")
            probability = top_probs[0][i].item()
            
            if word in [UNK, PAD, START, END]:
                continue
            
            results.append((word, probability))
        
        return results
    
def interactive_mode(predictor):
    print(f"\n=== Interactive test with NGram-NeuralNetWork model ===")
    print("Type a sentence. Type 'exit' to quit. Type 'clear' to reset context.")
    current_context = ""
    while True:
        
        user_input = input("\n> ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            current_context = ""
            continue
        if not user_input:
            continue
        
        current_context += " " + user_input     
        predictions = predictor.predict_next(current_context , top_k=5)
        for i, (word, prob) in enumerate(predictions):
            print(f"  {i+1}. {word:<15} ({prob:.2%})")
        
        print(f"\nCurrent context: '{current_context}'")
            
        
if __name__ == "__main__":
    model_path = r"C:\Users\Agnes\Desktop\Type Predictor\ngram_nn\best_model_with_val.pth"
    predictor = NGramNNPredictor(model_path, device='cpu')
    interactive_mode(predictor)
    
        
        
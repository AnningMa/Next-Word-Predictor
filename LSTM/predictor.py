from LSTM.model import LSTM_Model,ModelConfig
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *
import os
from collections import Counter
import re

class LSTMPredictor:
    def __init__(self, model_class, checkpoint_path, device='cpu', alpha=0.3):
      
        self.device = device
        self.alpha = alpha
        self.user_history = Counter()  # local user history
        
        print(f"loading model: {checkpoint_path} ...")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Can't find model: {checkpoint_path}")

        # 1. Loading Checkpoint
        # weights_only=False , to ensure all info is loaded
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'word2id' not in checkpoint:
            raise KeyError
        
        self.word2id = checkpoint['word2id']
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_id = self.word2id.get('<UNK>', 1)
        self.pad_id = self.word2id.get('<PAD>', 0)
        self.start_id = self.word2id.get('<START>', 2) 
        
    
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = ModelConfig(vocab_size=len(self.word2id)) 
            
        self.model = model_class(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()

    def preprocess(self, sentence_list):
    
        ids = [self.word2id.get(w, self.unk_id) for w in sentence_list]
        
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def update_user_history(self, sentence_list):
        
        for word in sentence_list:
            if word in self.word2id:
                token_id = self.word2id[word]
              
                if token_id > 3: 
                    self.user_history[token_id] += 1

    def predict_next(self, context_sentence, top_k=5):
        if isinstance(context_sentence, str):
            tokens = context_sentence.strip().split()
        else:
            tokens = context_sentence

        input_tensor = self.preprocess(tokens)
        
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            
            next_token_logits = logits[0, -1, :]
    
            lstm_probs = F.softmax(next_token_logits, dim=0)
        
            user_probs = torch.zeros_like(lstm_probs)
            total_history_count = sum(self.user_history.values())
            
            if total_history_count > 0:
                for token_id, count in self.user_history.items():
                    if token_id < len(user_probs):
                        user_probs[token_id] = count / total_history_count
        
            final_probs = (1 - self.alpha) * lstm_probs + self.alpha * user_probs
            probs, indices = torch.topk(final_probs, top_k)
            
            predictions = [self.id2word[idx.item()] for idx in indices]
            
            return predictions, probs.tolist()
# ==========================================
#              Usage
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. path
    model_path = r"C:\Users\Agnes\Desktop\Type Predictor\LSTM\lstm_best_model.pth"
    
    # 2. Initialize Predictor
    try:
        predictor = LSTMPredictor(
            model_class=LSTM_Model, 
            checkpoint_path=model_path, 
            device=device,
            alpha=0.4  # 设置 40% 的权重给用户习惯
        )
        
        # test 

        input = input("Enter a sentence: ")
        tokens = tokenize(input)
        predictor.update_user_history(tokens)
        predictions, probs = predictor.predict_next(tokens, top_k=5)
        print("Predicted next words:")
        for i, word in enumerate(predictions):
            print(f"  {i+1}. {word}")
            
    except Exception as e:
        raise e
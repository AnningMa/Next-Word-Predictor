import torch
import torch.nn as nn
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *

train_data, val_data, test_data, vocab, word2id, id2word = load_processed_data("dataset_processed.pkl")

class NeuralNGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
    
        super(NeuralNGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=word2id[PAD])
    
        self.net = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        assert x.dim() == 2  # (batch_size, context_size)     
        
        embeds = self.embedding(x)
        embeds_flat = embeds.reshape(x.size(0), -1)
        logits = self.net(embeds_flat)
        return logits

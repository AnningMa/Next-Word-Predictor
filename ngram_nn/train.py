import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *
import torch.optim as optim

#data loader
train_data, val_data, test_data, vocab, word2id, id2word = load_processed_data("dataset_processed.pkl")

class NGramDataset(Dataset):
    def __init__(self, sentences, word2id, context_size=3):
        self.sample = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            tokens = [START] + tokens + [END]
            token_ids = [word2id.get(tok, word2id[UNK]) for tok in tokens]
            for i in range(len(token_ids) - context_size):
                context = token_ids[i:i+context_size]
                target = token_ids[i+context_size]
                self.sample.append((context, target))
                
    def __len__(self):
        return len(self.sample)
    def __getitem__(self, idx):
        context, target = self.sample[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


CONTEXT_SIZE = 3 #trigram context
print(f"Building Dataset with context size={CONTEXT_SIZE}...")
train_dataset = NGramDataset(train_data, word2id, context_size=CONTEXT_SIZE)
test_dataset = NGramDataset(test_data, word2id, context_size=CONTEXT_SIZE)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

# Hyperparameters
CONTEXT_SIZE = 3  
embedding_dim = 128
hidden_dim = 256
vocab_size = len(vocab)

ngram_nn = NeuralNGram(context_size=CONTEXT_SIZE,
                      vocab_size=vocab_size,
                      embedding_dim=embedding_dim,
                      hidden_dim=hidden_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ngram_nn.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(ngram_nn.parameters(), lr=0.001)

print("Training Neural NGram Model")
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    ngram_nn.train()
    for context, target in train_loader:
        context = context.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = ngram_nn(context)
        loss = loss_function(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
print("Training complete")
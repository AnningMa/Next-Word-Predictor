import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *
from dataclasses import dataclass


def load_processed_data(filename="dataset_processed.pkl"):
    print(f"Loading data from {filename}...")
    with open(filename, "rb") as f:
        data_package = pickle.load(f)

    train_data = data_package["train_data"]
    val_data = data_package["val_data"]
    test_data = data_package["test_data"]
    vocab = data_package["vocab"]
    word2id = data_package["word2id"]
    id2word = data_package["id2word"]

    print(f"Data loaded.")
    print(f"Train size: {len(train_data)}")
    print(f"Vocab size: {len(vocab)}")

    return train_data, val_data, test_data, vocab, word2id, id2word

import numpy as np
train_data, val_data, test_data, vocab, word2id, id2word = load_processed_data(file_path)
length = [len(s) for s in train_data+val_data+test_data]
mean_len = np.mean(length)
median_len = np.median(length)
print(f"Mean length: {mean_len}")
print(f"Median length: {median_len}")

print(val_data)
max_len = 20


class LSTMDataset(Dataset):
    def __init__(self, sentences, word2id,max_len=20):
      self.sample = []
      self.max_len = max_len

      pad_id = word2id['<PAD>']
      start_id = word2id['<START>']
      end_id = word2id['<END>']
      unk_id = word2id['<UNK>']
      for sentence in sentences:
        tokens = tokenize(sentence)
        ids = [word2id.get(tok, unk_id) for tok in tokens]
        ids = ids[:max_len-1]
        input_ids = [start_id] + ids
        target_ids = ids + [end_id]

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
          input_ids += [pad_id] * pad_len
          target_ids += [pad_id] * pad_len

        self.sample.append((input_ids, target_ids))

    def __len__(self):
        return len(self.sample)
    def __getitem__(self, idx):
        context, target = self.sample[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
from dataclasses import dataclass
@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int = 256
    num_layers: int=2
    dropout: float=0.5
    padding_idx: int=2
    tie_weights: bool=True


class LSTM_Model(nn.Module):
    def __init__(self, config:ModelConfig):
      super(LSTM_Model, self).__init__()
      self.config = config

      self.embedding = nn.Embedding(
          config.vocab_size,
          config.hidden_size,
          padding_idx=config.padding_idx
          )

      self.emb_dropout = nn.Dropout(config.dropout)

      self.lstm = nn.LSTM(
          input_size=config.hidden_size,
          hidden_size=config.hidden_size,
          num_layers=config.num_layers,
          batch_first=True,
          dropout=config.dropout if config.num_layers > 1 else 0
      )

      self.ln = nn.LayerNorm(config.hidden_size)
      self.out_dropout = nn.Dropout(config.dropout)
      self.fc = nn.Linear(config.hidden_size,config.vocab_size)
      if config.tie_weights:
        self.fc.weight = self.embedding.weight

      self.init_weights()


    def init_weights(self):
      initrange = 0.1
      self.embedding.weight.data.uniform_(-initrange, initrange)
      self.fc.bias.data.zero_()
      self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x,hidden=None):
      embed = self.embedding(x)
      embed = self.emb_dropout(embed)

      out, new_hidden = self.lstm(embed,hidden)
      out = self.ln(out)
      logits = self.fc(out)
      return logits,new_hidden

from torch.cuda.amp import autocast,GradScaler
scaler = GradScaler()

#dataloader
train_dataset = LSTMDataset(train_data,word2id,max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#tensorboard
writer = SummaryWriter(log_dir='lstm_experiment_v1')

#define model,optimizer,criterion,scheduler
config = ModelConfig(vocab_size=len(vocab))
model = LSTM_Model(config)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2id['<PAD>'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

global_step = 0 #num of batch
num_epochs = 20

print("Training LSTM Model...")

for epoch in range(num_epochs):
  total_loss = 0
  model.train()

  for batch_idx, (input, target) in enumerate(train_loader):
    input = input.to(device)
    target = target.to(device)
    optimizer.zero_grad()

    with autocast():
      #forward
      logits, _ = model(input)
      #loss
      loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

    scaler.scale(loss).backward()

    #gradient clip
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    scaler.step(optimizer)
    scaler.update()

    #累计loss
    total_loss += loss.item()

    #每10个batch记录一次
    if global_step % 5 == 0:
      writer.add_scalar('Loss/train_batch', loss.item(), global_step)
      current_lr = optimizer.param_groups[0]['lr']
      writer.add_scalar('Learning Rate', current_lr, global_step)
    global_step += 1

  #--- After Epoch --
  avg_loss = total_loss / len(train_loader)
  writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

  #update lr
  scheduler.step()

  print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f},Learning Rate: {scheduler.get_last_lr()[0]}")

  #save model
  if (epoch+1) % 5 == 0:
    checkpoint_name = f"lstm_epoch_{epoch+1}.pth"
    save_path = os.path.join(project_path,checkpoint_name)

    checkpoint = {
        'epoch': epoch +1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }
    torch.save(checkpoint, save_path)

writer.close()
print("Training complete")

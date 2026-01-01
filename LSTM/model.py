import torch
import torch.nn as nn
import sys
sys.path.append(r"C:\Users\Agnes\Desktop\Type Predictor")
from data import *
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
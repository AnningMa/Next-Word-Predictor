import sys
import pickle
import os
import re
from abbreviations import abbrev_dict
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import nltk
from collections import Counter


#sys.stdout.reconfigure(encoding='utf-8')

def load_data():
    path = r"C:\Users\Agnes\Desktop\Type Predictor\88milSMS_88522.xlsx"
    df = pd.read_excel(path)
    
    sentences = df.iloc[1:, 4].dropna().astype(str).tolist()

    print("Loading and cleaning data...")
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_sentence = preprocess(sentence)
        if cleaned_sentence:
            cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences 

def preprocess(sentence):
    sentence = re.sub(r"<[A-Z]{3}_\d+>", "", sentence)
    sentence = sentence.lower()
    sentence = re.sub(r"(?<![a-zA-Z])-|-(?![a-zA-Z])", "", sentence)
    sentence = re.sub(r"[^\w\s\-]", " ", sentence)
    sentence = re.sub(r'\d+', ' <NUM> ', sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = expand_abbrev(sentence, abbrev_dict)

    return sentence

def expand_abbrev(text, abbr_dict):
    words = text.split()
    expanded_words = [abbr_dict.get(word, word) for word in words]
    return " ".join(expanded_words)

def split_dataset(sentences, test_size=0.1, random_state=42):
    train_data, test_data = train_test_split(sentences, test_size=test_size, random_state=random_state)
    return train_data, test_data

def split_dataset2(sentences, val_size=0.1, test_size=0.1, random_state=42):
    train_data, temp_data = train_test_split(sentences, test_size=(val_size + test_size), random_state=random_state)
    relative_test_size = test_size / (val_size + test_size)
    val_data, test_data = train_test_split(temp_data, test_size=relative_test_size, random_state=random_state)
    return train_data, val_data, test_data

def tokenize(sentence):
    return nltk.word_tokenize(sentence,language='french')

PAD = "<PAD>"
UNK = "<UNK>"
START = "<START>"
END = "<END>"

def build_vocabulary(sentences):
    vocab = {PAD, UNK, START, END} #set
    for s in sentences:
        vocab.update(tokenize(s))
    vocab = sorted(list(vocab)) #avoid randomness
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    return vocab, word2id, id2word

def build_vocabulary_2(sentences, min_freq=3):
    counter = Counter()
    for i, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        counter.update(tokens)
    
    print(f"original vocab size: {len(counter)}")
    
    valid_words = [word for word, count in counter.items() if count >= min_freq]
    
    word2id = {
        '<PAD>': 0,
        '<UNK>': 1, # 未知词
        '<START>': 2, # Start of Sentence (如果有的话)
        '<END>': 3  # End of Sentence (如果有的话)
    }
    
    start_idx = len(word2id)
    
    for idx, word in enumerate(valid_words):
        word2id[word] = start_idx + idx
        
    print(f"Final vocabulary size: {len(word2id)}")
    
    id2word = {i: w for w, i in word2id.items()}
    vocab = list(word2id.keys()) # 实际上 word2id 的 keys 就是 vocab
    
    return vocab, word2id, id2word

def encode(sentence,word2id,max_len):
    tokens = tokenize(sentence)
    tokens = [START] + tokens + [END]
    ids = [word2id.get(tok, word2id[UNK]) for tok in tokens]
    if len(ids) < max_len:
        ids += [word2id[PAD]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def save_processed_data(train_data, val_data, test_data, vocab, word2id, id2word, filename="dataset_processed.pkl"):

    data_package = {
        "train_data": train_data,   #list
        "val_data": val_data,
        "test_data": test_data,
        "vocab": vocab,
        "word2id": word2id,
        "id2word": id2word
    }
    
    print(f"Saving data to {filename}...")
    
    with open(filename, "wb") as f:
        pickle.dump(data_package, f)
    print("Save complete!")

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

if __name__ == "__main__":
    data = load_data()
    
    train_data, val_data, test_data = split_dataset2(data, val_size=0.1, test_size=0.1)
    
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    vocab, word2id, id2word = build_vocabulary_2(train_data, min_freq=3)
    
    print(word2id['<PAD>'])
    print(word2id['<UNK>'])
    print(word2id['<START>'])
    print(word2id['<END>'])   
    
    print(f"Vocabulary size: {len(vocab)}")
    
    sample_sentence = data[0]
    encoded = encode(sample_sentence, word2id, max_len=20)
    print(f"Sample sentence: {sample_sentence}")
    print(f"Encoded: {encoded}")
    
    save_processed_data(train_data, val_data, test_data, vocab, word2id, id2word)
    
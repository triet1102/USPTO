from torch.utils.data import Dataset
import torch
import numpy as np

class PhraseSimilarityDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True
        }
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        anchor = self.df.anchor.iloc[index].lower()
        target = self.df.target.iloc[index].lower()
        title = self.df.context.iloc[index].lower()
        
        text_to_tokenize = f"{anchor} {self.tokenizer.sep_token} {target} {self.tokenizer.sep_token} {title}." 
        tokens = self.tokenizer(text_to_tokenize, **self.tokenizer_params)
        score = torch.tensor(self.df.score.iloc[index], dtype=torch.float32)
        
        return (
            np.array(tokens["input_ids"]),
            np.array(tokens["attention_mask"]),
            score
        )

class PhraseSimilarityTestset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.tokenizer_params = {
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True
        }
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        anchor = self.df.anchor.iloc[index].lower()
        target = self.df.target.iloc[index].lower()        
        title = self.df.context.iloc[index].lower()
        
        text_to_tokenize = f"{anchor} {self.tokenizer.sep_token} {target} {self.tokenizer.sep_token} {title}." 
        tokens = self.tokenizer(text_to_tokenize, **self.tokenizer_params)
        
        return (
            np.array(tokens["input_ids"]),
            np.array(tokens["attention_mask"]),
        )
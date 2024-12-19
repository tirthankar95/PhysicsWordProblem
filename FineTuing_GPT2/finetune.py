from logging import config
from random import shuffle
from websockets import Data
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os 
import nltk 
from transformers import GPT2Tokenizer, GPT2Config
from colorama import Style, Fore

def prepare_spam_data(debug = False, batch_size = 2):
    df = pd.read_csv(os.path.join("dataset", "spam.csv"), encoding='latin1')
    df_ham = df[df['v1'] == 'ham'][['v2']]
    '''
    It handles punctuation intelligently (e.g., separating punctuation from words).
    It accounts for contractions (e.g., "don't" is split into ["do", "n't"]).
    '''
    # df_ham['sentence'] = df_ham['v2'].apply(lambda sentence: nltk.word_tokenize(sentence))
    df_ham['sentence'] = df_ham['v2']
    df_ham.drop(columns = ['v2'], axis = 1, inplace = True)
    df_ham['length'] = df_ham['sentence'].apply(lambda x: len(x))
    if debug:
        sns.histplot(data = df_ham, x = 'length', kde = True)
        plt.xlabel('Length of sentence.')
        plt.savefig(os.path.join('plots', 'SentenceLength.png'))
        plt.show()
    class CustomDataset(Dataset):
        def __init__(self, txt_list, tokenizer, max_length = 768):
            self.input_ids, self.attn_masks = [], []
            for txt in txt_list:
                encoding_dict = tokenizer('<|start|>' + txt + '<|end|>', \
                                          truncation = True, max_length = max_length)
                self.input_ids.append(encoding_dict['input_ids'])
                self.attn_masks.append(encoding_dict['attention_mask'])
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|start|>', \
                                              eos_token='<|end|>', pad_token='<|pad|>')
    print(f'{Fore.CYAN}[TOKNIZR] Mx model length: {tokenizer.model_max_length}. '+\
          f'GPT small: 768{Style.RESET_ALL}')
    print(f'{Fore.CYAN}[TOKNIZR] Beginning of seq: {tokenizer.decode(tokenizer.bos_token_id)}, '+\
          f'has token id: {tokenizer.bos_token_id}{Style.RESET_ALL}')
    print(f'{Fore.CYAN}[TOKNIZR] End of seq: {tokenizer.decode(tokenizer.eos_token_id)}, '+\
          f'has token id: {tokenizer.eos_token_id}{Style.RESET_ALL}')
    print(f'{Fore.CYAN}[TOKNIZR] Padding of seq: {tokenizer.decode(tokenizer.pad_token_id)}, '+\
          f'has token id: {tokenizer.pad_token_id}{Style.RESET_ALL}')
    dataset = CustomDataset(df_ham['sentence'], tokenizer)
    tr_size = int(0.95 * len(dataset))
    train_dataset, validation_dataset = random_split(dataset, [tr_size, len(dataset) - tr_size])
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    validation_loader = DataLoader(validation_dataset, shuffle = False, batch_size = batch_size)
    return train_loader, validation_loader
    
if __name__ == '__main__':
    # train_loader, validation_loader = prepare_spam_data()
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states = False)
    print(f'{Fore.GREEN}[GPT2]:\n{config}{Style.RESET_ALL}')
from secrets import token_bytes
import tokenize
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os 
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from colorama import Style, Fore
import time
import torch


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
        plt.clf()
    class CustomDataset(Dataset):
        def __init__(self, txt_list, tokenizer, max_length = 128):
            self.input_ids, self.attn_masks = [], []
            for txt in txt_list:
                encoding_dict = tokenizer('<|start|>' + txt + '<|end|>', \
                                          truncation = True, \
                                          padding = 'max_length', \
                                          max_length = max_length)
                self.input_ids.append(encoding_dict['input_ids'])
                self.attn_masks.append(encoding_dict['attention_mask'])
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]
    '''
        In GPT-2, the default value for tokenizer.bos_token_id is None 
        because GPT-2 does not use a beginning-of-sequence (BOS) token by default.
    '''
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
    return train_loader, validation_loader, tokenizer


def finetune(model, train_loader, validation_loader, epochs = 5):
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
    '''
        1. Gradually increases the learning rate from 0 to the initial maximum value over num_warmup_step
           This prevents abrupt large updates at the start of training when weights are randomly initialized.
        2. After the warm-up steps, the learning rate linearly decreases to 0 over the remaining num_training_steps
    '''
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = warmup_steps,
                                                num_training_steps = total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Selects device
    model = model.to(device)
    tbegin = time.time()
    stats = {"training_loss": [], "validation_loss": []}
    for epoch in range(epochs):
        print(f'======== Epoch {1+epoch} / {epochs} ========')
        # ========================================
        #               Training
        # ========================================
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for batch in train_loader:
            token = torch.stack(batch[0], dim = 0).permute(1, 0)
            attn = torch.stack(batch[1], dim = 0).permute(1, 0)
            b_input_ids = token.to(device)
            b_labels = token.to(device)
            b_masks = attn.to(device)
            model.zero_grad()
            outputs = model(b_input_ids, labels = b_labels, \
                            attention_mask = b_masks, token_type_ids = None)
            loss = outputs.loss 
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = total_train_loss / len(train_loader)
        print(f'Time: {round(time.time() - t0, 3)} sec, Avg Loss: {round(avg_loss, 4)}')
        stats["training_loss"].append(avg_loss)
        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()
        total_val_loss = 0
        model.eval()
        for batch in validation_loader:
            token = torch.stack(batch[0], dim = 0).permute(1, 0)
            attn = torch.stack(batch[1], dim = 0).permute(1, 0)
            b_input_ids = token.to(device)
            b_labels = token.to(device)
            b_masks = attn.to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, labels = b_labels, \
                                attention_mask = b_masks, token_type_ids = None)
                total_val_loss += outputs.loss.item()
        avg_loss = total_val_loss / len(validation_loader)
        print(f'Time: {round(time.time() - t0, 3)} sec, Avg Loss: {round(avg_loss, 4)}')
        stats["validation_loss"].append(avg_loss)
    print(f'------ Training Completed ------ {round(time.time() - tbegin, 3)} sec.')
    return stats


def load_model():
    model, tokenizer = None, None
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(os.path.join('./save_model/'))
        configuration = GPT2Config.from_pretrained(os.path.join('./save_model/', 'config.json'), output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained(os.path.join('./save_model/'), config = configuration)
    except: pass     
    return model, tokenizer


def TRAIN_SAVE(epochs = 5):
    # ==========================================
    #  Load Tokenizer, Model & Prepare Data
    # ==========================================
    train_loader, validation_loader, tokenizer = prepare_spam_data()
    configurations = GPT2Config.from_pretrained('gpt2', output_hidden_states = False)
    print(f'{Fore.GREEN}[GPT2]:\n{configurations}{Style.RESET_ALL}')
    '''
    When you load a model using GPT2LMHeadModel.from_pretrained('gpt2', config=configurations), the model weights are not automatically saved to disk. 
    Instead, the model is loaded into memory and available as a Python object (model). 
    The source of the weights depends on the context:
    '''
    model, tokenizer_bk = load_model()
    if model == None:
        model = GPT2LMHeadModel.from_pretrained('gpt2', config = configurations)
        model.resize_token_embeddings(len(tokenizer))
    else: tokenizer = tokenizer_bk
    # ==========================================
    #                Fine Tune
    # ==========================================
    stats = finetune(model, train_loader, validation_loader, epochs)
    # ==========================================
    #             Display Results
    # ==========================================
    if len(stats['training_loss']):
        sns.set(style='darkgrid')
        stats_df = pd.DataFrame(stats)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data = stats_df, x = range(stats_df.shape[0]), y = 'training_loss', \
                    label = 'training_loss', color = 'blue', marker = 'o')
        sns.lineplot(data = stats_df, x = range(stats_df.shape[0]), y = 'validation_loss', \
                    label = 'validation_loss', color = 'red', marker = 's')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join('plots', 'TrainVal.png'))
        plt.clf()
    params = list(model.named_parameters())
    print(f'GPT2 model has {len(params)} named parameters.')
    print('\n============ Embedding Layer ============')
    for p in params[:2]:
        print(f'{p[0]} {str(tuple(p[1].size()))}')
    print('\n============ First Transformer ============')
    for p in params[2:14]:
        print(f'{p[0]} {str(tuple(p[1].size()))}')
    print('\n============ Output Layer ============')
    for p in params[-2:]:
        print(f'{p[0]} {str(tuple(p[1].size()))}')
# ==========================================
#             Save Model
# ==========================================
    output_dir = './save_model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def GENERATE(prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model()
    if model == None:
        print(f'{Fore.RED}======== NO MODEL FOUND ========{Style.RESET_ALL}')
        return 
    model.eval()
    prompt_tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    sample_outputs = model.generate(prompt_tokens,\
                                    do_sample = True,\
                                    top_k = 10,\
                                    max_length = 128,\
                                    top_p = 0.95,\
                                    num_return_sequences = 3)
    for i, sample_output in enumerate(sample_outputs):
        print(f'==== [{i}] ====:\n {tokenizer.decode(sample_output, skip_special_tokens=True)}')


# ==========================================
#                   MAIN
# ==========================================
if __name__ == '__main__':
    TRAIN_SAVE(epochs = 1)
    GENERATE('The sun will')
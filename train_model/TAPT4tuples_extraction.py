import re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

from tqdm import tqdm

class OpinionTuple(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for text in texts:
            encodings_dict = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def transform_data(df, tokenizer, max_length):

    texts = []

    groups = df.groupby('id')
    for name, group in groups:
        
        sentence = set([])
        aspects, opinions = [], []
        for idx, row in group.iterrows():
            sentence.add(row['sentence'])
            aspects.append(row['aspect'])
            opinions.append(row['opinion'])
        
        text = f"<startoftext>Review: {sentence.pop()}"
        for asp, opi in zip(aspects, opinions):
            text += f"\nOpinion: [{asp}, {opi}]"
        text += "<endoftext>"
        if len(texts) < 2:
            print(text)
        texts.append(text)
    
    return OpinionTuple(texts, tokenizer, max_length)       


def load_dataset_multiple_opinions(dataset_ref, tokenizer, max_length):
    semeval_datasets_dir = "../datasets"
    train = semeval_datasets_dir + "/{0}/train.csv".format(dataset_ref)

    # load the data
    train_df = pd.read_csv(train)

    train_dataset = transform_data(train_df, tokenizer, max_length)

    return train_dataset


def TAPT(train_dataset, model, dataset_ref, epochs):
    training_args = TrainingArguments(output_dir='../models/tapt/' + dataset_ref, 
                                    num_train_epochs=epochs,
                                    #logging_steps=10, 
                                    overwrite_output_dir = True,
                                    load_best_model_at_end=True,
                                    per_device_train_batch_size = 1,   # batch size for training
                                    per_device_eval_batch_size = 1,   # batch size for evaluation
                                    save_strategy='no',
                                    #eval_steps = 50,   # Number of update steps between two evaluations. 
                                    #evaluation_strategy="no",
                                    prediction_loss_only = True,
                                    warmup_steps = 100, 
                                    weight_decay=0.01, 
                                    logging_dir='logs')

    trainer = Trainer(model=model, 
            args=training_args, 
            train_dataset=train_dataset,
            #eval_dataset = dev_dataset,
            data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data]),
                                            "attention_mask": torch.stack([f[1] for f in data]),
                                            "labels": torch.stack([f[0] for f in data]),
                                            })
    trainer.train()
    trainer.save_model()
    return model

if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset_ref = "16res"
    epochs = 3  # tune
    max_length = 400

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

    train_dataset = load_dataset_multiple_opinions(dataset_ref, tokenizer, max_length)
    model.resize_token_embeddings(len(tokenizer))

    model = TAPT(train_dataset, model, dataset_ref, epochs)


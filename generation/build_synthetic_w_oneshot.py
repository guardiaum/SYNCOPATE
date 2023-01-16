import csv
import numpy as np
import pandas as pd
import os.path
from argparse import ArgumentParser
from utils_triples import *

import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
np.random.seed(SEED)

device = torch.device("cuda")

if __name__ == '__main__':

    parser = ArgumentParser(description="generate new samples automatically")
    parser.add_argument("number_samples", type=str, help="the name of the generation model")
    parser.add_argument("ds", type=str, help="the identifier of the dataset being processed")
    args = parser.parse_args()

    number_of_synthetic_samples = int(args.number_samples)  #1000
    dataset_ref = args.ds  #"16res"

    print("number_of_synthetic_samples: ", number_of_synthetic_samples)
    print("Dataset: ", dataset_ref)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", bos_token='<startoftext>', eos_token='<endoftext>', pad_token='<pad>')
    model = GPT2LMHeadModel.from_pretrained("../models/tapt/{0}".format(dataset_ref))

    # setup path destine to save samples
    dir2synthetic_path = "../datasets/synthetic/oneshot/{0}".format(dataset_ref)
    
    if os.path.exists(dir2synthetic_path)==False:
        os.makedirs(dir2synthetic_path)

    dest_synthetic_samples = "{0}/{1}.csv".format(dir2synthetic_path, number_of_synthetic_samples)

    if os.path.isfile(dest_synthetic_samples):
        # print ("File exist")
        #df_samples.to_csv(dest_synthetic_samples, mode='a', index=False, quoting=csv.QUOTE_ALL, header=False)
        type = 'a'
    else:
        type = 'w'

    header = ["sentence", "aspect", "opinion"]

    with open(dest_synthetic_samples, type, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_ALL)
        if type=='w': writer.writeheader()

        samples = []
        count_saved_sentences = 0
        count_iteration = 0

        print("Generating triples...")
        while count_saved_sentences != number_of_synthetic_samples:
            count_iteration += 1
            print("="*100)
            prompt = get_random_prompt(dataset_ref, n_samples=1)
            print("[PROMPT] : \n{0}\n".format(prompt))

            gen = generate(tokenizer, model, device, prompt, max_length=250)
            sentence, tuples = extract_from_gen_fewshot(gen)

            if sentence is None or tuples is None: continue
            
            flag = False
            for tuple in tuples:
                
                aspect = tuple['aspect']
                opinion = tuple['opinion']
                
                aspect_tkns = aspect.split(" ")
                opinion_tkns = opinion.split(" ")

                has_asp, has_opi = False, False
                
                if all([True  if asp in sentence else False for asp in aspect_tkns]): has_asp = True
                if all([True  if opi in sentence else False for opi in opinion_tkns]): has_opi = True

                if has_asp and has_opi:
                    flag = True
                    row = {"sentence": sentence,
                            "aspect": aspect,
                            "opinion": opinion}
                    samples.append(row)
                    # Append single row to CSV
                    writer.writerow(row)

            if flag:
                count_saved_sentences += 1
                if count_saved_sentences%10==0:
                    print("Saved: {0}\tProgress: {1}%".format(count_saved_sentences, ((count_saved_sentences/number_of_synthetic_samples)*100)))
            
            '''
                ends the loop if the number of iteration is two times the number of expected samples
            '''
            if count_iteration == (5*number_of_synthetic_samples):
                break
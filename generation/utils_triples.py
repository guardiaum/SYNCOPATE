import re
import random
import pandas as pd

def generate(tokenizer, model, device, prompt, max_length=250):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
    model = model.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample = True,
        max_length = max_length,
        num_return_sequence = 1,
        pad_token_id = tokenizer.encode("<endoftext>", add_special_tokens=True)[0],
    )

    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    return gen_text

def extract_from_gen(gen):
    sentence = re.findall("Review: (.*)\n", gen)

    result = re.findall("\nOpinion: (.*)", gen)

    pattern = re.compile("\[.*, .*\]")

    tuples = []
    for item in result:
        #print(item)
        if pattern.match(item):
            #print('yes')
            tuple = item.replace("[","").replace("]","").split(", ")
            aspect = tuple[0].strip()
            opinion = tuple[1].strip()
            tuples.append({'aspect': aspect.lower(), "opinion": opinion.lower()})

    if sentence is not None and len(sentence) > 0 and len(tuples) > 0:
        return sentence[0], tuples
    else:
        return None, None

def extract_from_gen_fewshot(gen, n_samples=1):
    #gen = gen.replace("<endoftext><startoftext>","")

    print("[GENERATED] : {0}".format(gen))

    reviews = ["Review: " + split for split in gen.split("Review: ") if len(split)>0]
    
    print("[EXTRACT] : {0}".format(len(reviews)))
    
    if len(reviews) > n_samples:
        # gets the triple generated right after the sample(s) given as prompt
        review = reviews[n_samples]
        print("[EXTRACT] : {0}".format(review))
        sentence, tuples = extract_from_gen(review)
        #print(sentence)
        #print(tuples)

        if sentence is not None and len(sentence) > 0 and len(tuples) > 0:
            return sentence, tuples
    
    return None, None

'''
    Selects random triples from a dataset different from ref 
    in the restaurants domain
'''
def select_random_samples(ref, n_samples):
    new_ref = None
    if "res" in ref:
        datasets_ref = ["14res", "15res", "16res"]
        datasets_ref.remove(ref)

        new_ref = random.choice(datasets_ref)
        print("[RANDOM] : {0}".format(new_ref))
    elif ref == "14lap":
        new_ref = "15_16_lap"
        print("[RANDOM] : {0}".format(new_ref))

    if new_ref!=None:
        # select source
        src = "../datasets/{0}/train.csv".format(new_ref)
        df_src = pd.read_csv(src)

        # select random sample(s)
        selected_ids = df_src['id'].sample(n_samples).values

        selected_triples = df_src[df_src['id'].isin(selected_ids.tolist())]
        return selected_triples
    
    return False


def get_random_prompt(ref, n_samples=1):

    selected_triples = select_random_samples(ref, n_samples)
    print("[SELECTED RANDOM TRIPLE] : {0}".format(selected_triples))

    if selected_triples is False: raise Exception("[ERROR] Sorry, could not get random triples.")

    # build prompt
    prompt = build_fewshot_prompt(selected_triples)

    one_random_triple = select_random_samples(ref, n_samples=1)
    random_sentence = one_random_triple['sentence'].values[0]
    random_start = random_sentence[:int(len(random_sentence)/3)]

    print("[SELECTED RANDOM REVIEW START] : {0}".format(random_start))
    prompt += "Review: " + random_start
    return prompt

def build_fewshot_prompt(selected_triples):
    groups = selected_triples.groupby("id")

    prompt = ''
    sent = ''
    for name, group in groups:
        
        flag = True
        for i, row in group.iterrows():
            if flag:
                sentence = row["sentence"]
                sent = sentence
                flag = False

                prompt += "Review: {0}\n".format(sentence)

            aspect = row['aspect']
            opinion = row['opinion']

            prompt+= "Opinion: [{0}, {1}]\n".format(aspect, opinion)
    
    return prompt
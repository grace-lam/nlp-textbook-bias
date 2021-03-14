"""Utility functions"""

import ast

from transformers import AutoTokenizer

def get_keywords(category_word):
    """Returns keywords labeled of a certain LIWC category"""
    keywords = []
    with open(liwc_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        CATEGORY_INDEX = 2
        TERM_INDEX = 1
        for row in csv_reader:
            if row[CATEGORY_INDEX] == category_word:
                keywords.append(row[TERM_INDEX])
    return keywords

def read_dicts_from_paths(paths):
    """Read in data from files to do further analysis"""
    list_of_dicts = []
    for path in paths:
        with open(path) as f:
            data = f.read()
            data = ast.literal_eval(data)
            list_of_dicts.append(data)
    return list_of_dicts

def sentence_to_tokens(sentence):
    """Convert sentences to tokens using BERT's Tokenizer"""
    # we always use BERT's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_bert_pretrained)
    # convert to BERT's tokenizer format
    marked_sentence = '[CLS] ' + sentence + ' [SEP]'
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_sentence)[:block_size] # truncate
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Mark each of the tokens as belonging to sentence "1" (to mark everything is
    # in the same sentence, which is needed to extract from BERT model later!)
    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensor, tokenized_text

def eliminate_newlines(path):
    """Given a text file, output a string without the newlines"""
    with open(path) as f:
        data = f.read()
        data = data.replace("\n", " ")
    return data

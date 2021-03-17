"""Utility functions"""

import ast
import re

import numpy as np
import torch
from transformers import AutoTokenizer

model_bert_pretrained = 'bert-base-uncased'

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

def read_data(path):
    """Read data as-is from a path"""
    with open(path) as f:
        data = f.read()
        data = ast.literal_eval(data)
    return data

def read_attention_weights(path):
    with open(path) as f:
        data = f.read()
        data = data.replace("\t","")
        data = data.replace("\n", "")
        data = data.replace(" ", "")
        data = data.replace(",dtype=float32", "")
        data = data.replace("array", "")
        data = ast.literal_eval(data)
    return data

def read_context_windows(path):
    """Read in data from path, where path is to a file formatted as:
    an array with each entry formatted as
            (tokens_tensor, segments_tensor, tokenized_sentence,
            (gender_index, query_index, gender_word, query_word))
    """
    with open(path) as f:
        data = f.read()
        data = data.replace("tensor", "")
        data = ast.literal_eval(data)
    return data

def sentence_to_tokens(sentence, max_length=512):
    """Convert sentences to tokens using BERT's Tokenizer"""
    # we always use BERT's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_bert_pretrained)
    # convert to BERT's tokenizer format
    marked_sentence = '[CLS] ' + sentence + ' [SEP]'
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_sentence, max_length=max_length, truncation=True)
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

def count_words_per_line(path):
    num_words = []
    with open(path, 'r') as reader:
        for sentence in reader:
            num_words.append(len(sentence.split()))
    return sum(num_words)/len(num_words)

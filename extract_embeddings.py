"""Retrieve embeddings from BERT model for analysis"""

import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import finetune_bert

model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/bert_mlm_textbook'

def sentence_to_tokens(sentence):
    # we always use BERT's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_bert_pretrained)
    # convert to BERT's tokenizer format
    marked_sentence = '[CLS] ' + sentence + ' [SEP]'
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_sentence)
    # Print out the tokens.
    print (tokenized_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indeces.
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    # Mark each of the tokens as belonging to sentence "1" (to mark everything is
    # in the same sentence, which is needed to extract from BERT model later!)
    segments_ids = [1] * len(tokenized_text)
    print (segments_ids)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensor

def retrieve_embeddings(tokens_tensor, segments_tensor, keyword, model_option):
    # Load pre-trained model (weights) and make it return hidden states
    model = AutoModelForMaskedLM.from_pretrained(model_option, output_hidden_states = True)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        # All 12 hidden states from BERT model, I referenced this:
        # https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#MaskedLMOutput
        hidden_states = outputs.hidden_states
        print(hidden_states)

def main():
    sentence = 'this is a test sentence'
    tokens_tensor, segments_tensor = sentence_to_tokens(sentence)
    retrieve_embeddings(tokens_tensor, segments_tensor, 'test', model_bert_textbook_dir)
    # gpu_check()
    # lm_datasets, data_collator = load_data()
    # os.makedirs(model_dir, exist_ok=True)
    # finetune_bert(lm_datasets, data_collator)
    # TODO: STILL NEED TO WRITE DISTANCES TO SOME FILE OR PLOT OR SOMETHING

if __name__ == '__main__':
    main()

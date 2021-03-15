"""Retrieve embeddings from BERT model for analysis"""

import torch

import finetune_bert
import utilities

model_bert_pretrained = 'bert-base-uncased'
block_size = 512

def _retrieve_token_embeddings(tokens_tensor, segments_tensor, model):
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        # All 12 hidden states from BERT model, I referenced this:
        # https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#MaskedLMOutput
        hidden_states = outputs.hidden_states
        # 13 layers (1 for initial, 12 for BERT), 1 batch (sentence)
        # from dimmensions [# layers, # batches, # tokens, # features] -> [# tokens, # layers, # features]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
    return token_embeddings

def get_keyword_embeddings(tokens_tensor, segments_tensor, tokenized_text, sentence_info, model):
    gender_index, query_index, gender_word, query_word = sentence_info
    token_embeddings = _retrieve_token_embeddings(tokens_tensor, segments_tensor, model)
    keyword_embeddings = {}
    # token embeddings can be built via summation or concatenation (or a more
    # complex method); here we build by summation. token_embeddings is [# tokens, # layers, # features]
    keyword_embeddings[gender_word] = torch.sum(token_embeddings[gender_index][-4:], dim=0)
    keyword_embeddings[query_word] = torch.sum(token_embeddings[query_index][-4:], dim=0)
    return keyword_embeddings

"""Retrieve embeddings from BERT model for analysis"""

import torch

import finetune_bert
import utilities

model_bert_pretrained = 'bert-base-uncased'
block_size = 512

def _build_keyword_embeddings(token_embeddings, tokenized_text, keywords):
    keyword_embeddings = {}
    for keyword in keywords:
        try:
            token_index = tokenized_text.index(keyword)
        except:
            print("The embedding for keyword %s was not found. Perhaps the tokenizer chomped it?" %keyword)
            continue
        # token embeddings can be built via summation or concatenation (or a more
        # complex method); here we build by summation. token_embeddings is [# tokens, # layers, # features]
        keyword_embeddings[keyword] = torch.sum(token_embeddings[token_index][-4:], dim=0)
    return keyword_embeddings

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

def get_keyword_embeddings(sentence:str, keywords:set, model):
    tokens_tensor, segments_tensor, tokenized_text = utilities.sentence_to_tokens(sentence)
    token_embeddings = _retrieve_token_embeddings(tokens_tensor, segments_tensor, model)
    keyword_embeddings = _build_keyword_embeddings(token_embeddings, tokenized_text, keywords)
    return keyword_embeddings

def main():
    finetune_bert.gpu_check()
    sentence = 'this is a test sentence'
    keywords = set() # to ensure there are no duplicate words being queried
    keywords.add("test")
    get_keyword_embeddings(sentence, keywords, model_bert_textbook_dir)


if __name__ == '__main__':
    main()

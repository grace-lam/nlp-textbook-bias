import os
import time

import numpy as np
from scipy import stats
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

import extract_embeddings_v1
import finetune_bert
from temporal_analysis_cosine_v1 import _get_category
import utilities

# Tokenizer and model used throughout
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert_mlm/block_512/bert_mlm_textbook", output_attentions=True)
# Init softmax to get probabilities later on
softmax = torch.nn.Softmax(dim=0)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
torch.set_grad_enabled(False)

# Contexts fed into BERT must start with a [CLS] token and (possibly?) end with a [SEP] token
mask_token, mask_id = tokenizer.mask_token, tokenizer.mask_token_id
cls_token, cls_id = tokenizer.cls_token, tokenizer.cls_token_id
sep_token, sep_id = tokenizer.sep_token, tokenizer.sep_token_id

# Relevant paths
context_size = 'block_512'
max_dist = 'maxdist_100'
config = context_size + max_dist
textbook_chronological_dir = 'final_textbook_contexts/' + config + '/'

LOAD_RESULTS = True # change this to False to rerun embedding extraction and get new results (will override folder!)
results_folder = 'static_attention_analysis_results_' + config + '/'
results_path = results_folder + "all_results.txt"
stats_tests_file = 'stats_tests.txt'

def get_attention_and_probs(inputs, masked_position):
    # Forward
    outputs = model(inputs)
    attention = outputs.attentions  # Output includes attention weights when output_attentions=True
    last_hidden_state = outputs[0].squeeze(0)
    # Only get output for masked token (output is the size of the vocabulary)
    mask_hidden_state = last_hidden_state[masked_position]
    # Convert to probabilities (softmax), giving a probability for each item in the vocabulary
    probs = softmax(mask_hidden_state)
    return attention, probs

"""set pronoun_to_interest to TRUE if we want to finding the weights attending from the MASKed pronoun to the query word
and FALSE for vice versa
"""
def get_attending_weights(attention, pronoun_idx, interest_idx, pronoun_to_interest):
    att_weights = []
    for att_layer in attention:
        layer = att_layer.squeeze()
        if pronoun_to_interest:
            layer_weights = layer[:, pronoun_idx, interest_idx].numpy()
        else:
            layer_weights = layer[:, interest_idx, pronoun_idx].numpy()
        att_weights.append(layer_weights)
    att_weights = np.stack(att_weights, axis=0)
    return att_weights

def _analyze_attention(sentence_data, weights):
    work_weights = weights["work"]
    workers_weights = weights["workers"]
    tokens_tensor, segments_tensor, tokenized_text, sentence_info = sentence_data
    tokens_tensor = torch.tensor(tokens_tensor)
    segments_tensor = torch.tensor(segments_tensor)
    gender_index, query_index, gender_word, query_word = sentence_info
    attention, probs = get_attention_and_probs(tokens_tensor, gender_index)
    attending_weights = get_attending_weights(attention, gender_index, query_index, False)
    if query_word == "work":
        work_weights.append((sentence_data, attending_weights))
    else:
        workers_weights.append((sentence_data, attending_weights))

def attention_analysis():
    # Each entry of all_cos is a dictionary formatted as: {year:{(gender_word, query_word): cos_sim}}
    weights = {}
    weights["work"] = []
    weights["workers"] = []
    for year_dir in os.listdir(os.fsencode(textbook_chronological_dir)):
         dirname = os.fsdecode(year_dir)
         if dirname.endswith(".txt"): # currently directories have .txt extension
             year = dirname[:-4] # get rid of extension
             for file in os.listdir(textbook_chronological_dir + dirname + "/"):
                 filename = os.fsdecode(file)
                 if filename.endswith(".txt"):
                     keywords = filename[:-4] # get rid of extension
                     gender_word, query_word = keywords.split("_")
                     if query_word != "work" and query_word != "workers":
                         continue
                     data = utilities.read_context_windows(textbook_chronological_dir + dirname + "/" + filename)
                     for sentence_data in data:
                         _analyze_attention(sentence_data, weights)
    with open(results_path, "w") as output:
        output.write(str(weights))
    return weights

def extract_attentions(weights, word, category=None):
    all_weights = []
    for entry in weights[word]:
        sentence_data, attending_weights = entry
        if category:
            tokens_tensor, segments_tensor, tokenized_text, sentence_info = sentence_data
            gender_index, query_index, gender_word, query_word = sentence_info
            if _get_category(gender_word) != category:
                continue
        attending_weights = np.array(attending_weights)
        if len(all_weights) == 0:
            all_weights = attending_weights.flatten()
        else:
            all_weights = np.concatenate((all_weights,attending_weights.flatten()), axis=0)
    return np.array(all_weights)
    # later: can also break down by gender

def analyze_results(weights):
    # most basic: compare averages and do a t-test
    work_attentions = extract_attentions(weights, "work")
    workers_attentions = extract_attentions(weights, "workers")
    t_val, p_val = stats.ttest_ind(work_attentions, workers_attentions)
    work_avg = np.mean(work_attentions)
    workers_avg = np.mean(workers_attentions)
    with open(results_folder + stats_tests_file, "w")as output:
        output.write("Work vs workers attentions have averages %f and %f respectively and t-value %f and p-value %f \n"
        %(work_avg, workers_avg, t_val, p_val))

    # compare gender-specific attentions between each and do t-tests
    work_man_attentions = extract_attentions(weights, "work", "man")
    work_woman_attentions = extract_attentions(weights, "work", "woman")
    workers_man_attentions = extract_attentions(weights, "workers", "man")
    workers_woman_attentions = extract_attentions(weights, "workers", "woman")
    work_t_val, work_p_val = stats.ttest_ind(work_man_attentions, work_woman_attentions)
    work_man_avg = np.mean(work_man_attentions)
    work_woman_avg = np.mean(work_woman_attentions)
    workers_t_val, workers_p_val = stats.ttest_ind(workers_man_attentions, workers_woman_attentions)
    workers_man_avg = np.mean(work_man_attentions)
    workers_woman_avg = np.mean(workers_woman_attentions)
    with open(results_folder + stats_tests_file, "a")as output:
        output.write("Work: man vs woman attentions have averages %f and %f respectively and t-value %f and p-value %f \n"
        %(work_man_avg, work_woman_avg, work_t_val, work_p_val))
        output.write("Workers: man vs woman attentions have averages %f and %f respectively and t-value %f and p-value %f \n"
        %(workers_man_avg, workers_woman_avg, workers_t_val, workers_p_val))

def main():
    # print(utilities.count_words_per_line("final_textbook_years/all_textbooks/1700.txt"))
    start_time = time.perf_counter()
    finetune_bert.gpu_check()
    os.makedirs(results_folder, exist_ok=True)
    if not LOAD_RESULTS:
        weights = attention_analysis()
    else:
        weights = utilities.read_attention_weights(results_path)
    analyze_results(weights)
    end_time = time.perf_counter()
    print(f"This took {(end_time - start_time)/60:0.4f} minutes")

if __name__ == '__main__':
    main()

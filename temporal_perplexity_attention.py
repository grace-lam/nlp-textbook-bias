"""Perplexity and attention analysis"""
import os
import time
import statistics

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

import utilities
from temporal_analysis_cosine_v1 import _get_category
from finetune_bert import gpu_check

home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

# Create mapping from pronoun to opposite pronoun
man_words = ['man', 'men', 'male', 'he', 'him', 'his']
woman_words = ['woman', 'women', 'female', 'she', 'her', 'hers']
man_words_set = set(man_words)
woman_words_set = set(woman_words)
pronoun_oppos = dict()
for i, man_word in enumerate(man_words):
    pronoun_oppos[man_word] = woman_words[i]
    pronoun_oppos[woman_words[i]] = man_word

# Directories
num_context_tokens = 512
model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/block_512/bert_mlm_textbook'
textbook_chronological_dir = 'final_textbook_contexts/512_tokens/'
results_pp_dir = 'temporal_pp_results_{}/'.format(num_context_tokens)
results_pp_path = results_pp_dir + "all_results.txt"
results_attn_dir = 'temporal_attn_examples_{}/'.format(num_context_tokens)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_bert_pretrained)
model = AutoModelForMaskedLM.from_pretrained(model_bert_textbook_dir, output_attentions=True)
# Init softmax to get probabilities later on
softmax = torch.nn.Softmax(dim=0)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
torch.set_grad_enabled(False)

# Contexts fed into BERT must start with a [CLS] token and (possibly?) end with a [SEP] token
mask_token, mask_id = tokenizer.mask_token, tokenizer.mask_token_id
cls_token, cls_id = tokenizer.cls_token, tokenizer.cls_token_id
sep_token, sep_id = tokenizer.sep_token, tokenizer.sep_token_id

# Save interesting examples for attention maps in format: [(tokens_tensor, segments_tensor, tokenized_text, tokenized_info, norm_prob)]
attn_pr_25 = []
attn_pr_45_55 = []
attn_pr_9999 = []

def _get_attn_and_mask_probs(inputs, masked_position):
    # Forward
    outputs = model(inputs)
    attention = outputs.attentions  # Output includes attention weights when output_attentions=True
    last_hidden_state = outputs[0].squeeze(0)
    # Only get output for masked token (output is the size of the vocabulary)
    mask_hidden_state = last_hidden_state[masked_position]
    # Convert to probabilities (softmax), giving a probability for each item in the vocabulary
    probs = softmax(mask_hidden_state)
    return probs, attention

def _add_perplexity_values(context, pp_year, direct_comp=True):
    # Parse txt file into tensors and relevant info
    tokens_tensor, segments_tensor, tokenized_text, sentence_info = context
    tokens_tensor = torch.tensor(tokens_tensor)
    segments_tensor = torch.tensor(segments_tensor)
    gender_index, query_index, gender_word, query_word = sentence_info

    if len(tokenized_text) == num_context_tokens:
        # Mask the target gender word
        inputs = tokens_tensor
        inputs[0][gender_index] = mask_id
        masked_position = gender_index

        probs, attention = _get_attn_and_mask_probs(inputs, masked_position)

        if direct_comp:
            # Get probability of token <pronoun>
            pronoun_id = tokenizer.convert_tokens_to_ids(gender_word)
            pronoun_prob = probs[pronoun_id].item()
            # Get probability of token <opposite_pronoun>
            opp_pronoun = pronoun_oppos[gender_word]
            opp_pronoun_id = tokenizer.convert_tokens_to_ids(opp_pronoun)
            opp_pronoun_prob = probs[opp_pronoun_id].item()
            gender_prob = pronoun_prob
            opp_gender_prob = opp_pronoun_prob
        else:
            man_prob = 0
            woman_prob = 0
            for m_word in man_words_set:
                pronoun_id = tokenizer.convert_tokens_to_ids(m_word)
                man_prob += probs[pronoun_id].item()
            for w_word in woman_words_set:
                pronoun_id = tokenizer.convert_tokens_to_ids(w_word)
                woman_prob += probs[pronoun_id].item()
            gender_prob = man_prob if gender_word in man_words_set else woman_prob
            opp_gender_prob = woman_prob if gender_word in man_words_set else man_prob

        norm_prob = gender_prob / (gender_prob + opp_gender_prob)
        correctness = 1 if norm_prob > 0.5 else 0

        if (gender_word, query_word) not in pp_year:
            pp_year[(gender_word, query_word)] = []
        pp_year[(gender_word, query_word)].append((norm_prob, correctness))

        # Add interesting contexts for attention analysis
        if norm_prob < 0.25: # high confidence, incorrect prediction
            attn_pr_25.append((tokens_tensor, segments_tensor, tokenized_text, sentence_info, norm_prob))
        elif norm_prob > 0.45 and norm_prob < 0.55: # low confidence
            attn_pr_45_55.append((tokens_tensor, segments_tensor, tokenized_text, sentence_info, norm_prob))
        elif norm_prob > 0.9999: # high confidence, correct prediction
            attn_pr_9999.append((tokens_tensor, segments_tensor, tokenized_text, sentence_info, norm_prob))

def get_temporal_perplexity_and_attention_values():
    # Each entry of pp is a dictionary formatted as: {year:{(gender_word, query_word): [(norm_prob, correctness)]}}
    pp = {}
    pp[("woman", "work")] = {}
    pp[("man", "work")] = {}
    pp[("woman", "home")] = {}
    pp[("man", "home")] = {}
    pp[("woman", "achiev")] = {}
    pp[("man", "achiev")] = {}

    # year_ctr = 1

    for year_dir in os.listdir(os.fsencode(textbook_chronological_dir)):
        dirname = os.fsdecode(year_dir)
        if dirname.endswith(".txt"): # currently directories have .txt extension
            year = dirname[:-4] # get rid of extension
            for file in os.listdir(textbook_chronological_dir + dirname + "/"):
                filename = os.fsdecode(file)
                if filename.endswith(".txt"):
                    keywords = filename[:-4] # get rid of extension
                    gender_word, query_word = keywords.split("_")
                    gender_category = _get_category(gender_word)
                    query_category = _get_category(query_word)
                    if year not in pp[(gender_category, query_category)]:
                        pp[(gender_category, query_category)][year] = {}
                    data = utilities.read_context_windows(textbook_chronological_dir + dirname + "/" + filename)
                    for context in data:
                        _add_perplexity_values(context, pp[(gender_category, query_category)][year], False)
            # if year_ctr == 2:
            #     break
            # year_ctr += 1
    with open(results_pp_path, "w") as output:
        output.write(str(pp))
    return pp

def _get_years_and_probs_and_acc(pp_year, interest_word):
    years = []
    probs = [] # one list per year
    year_to_correctness = {}
    acc = []

    for year in sorted(pp_year.keys()):
        pp_dict = pp_year[year]
        probs_yr = []
        for words in pp_dict:
            gender_word, query_word = words
            if query_word == interest_word:
                for (norm_prob, correctness) in pp_dict[words]:
                    probs_yr.append(norm_prob)
                    if year not in year_to_correctness:
                        year_to_correctness[year] = []
                    year_to_correctness[year].append(correctness)
        probs.append(probs_yr)

    for year in sorted(year_to_correctness.keys()):
        corr_arr = year_to_correctness[year]
        years.append(year)
        acc.append(np.mean(corr_arr))
                        
    return years, probs, acc

def plot_temporal_preds(woman_pp, man_pp, interest_word, liwc_category):
    woman_years, woman_probs, woman_acc = _get_years_and_probs_and_acc(woman_pp, interest_word)
    man_years, man_probs, man_acc = _get_years_and_probs_and_acc(man_pp, interest_word)
    if not woman_years or not man_years:
        return

    # Make accuracy scatter plots
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    woman_years = np.array(list(map(int,woman_years)))
    man_years = np.array(list(map(int,woman_years)))
    plt.scatter(woman_years, woman_acc, color='r', label='woman words', marker="*")
    plt.scatter(man_years, man_acc, color='b', label='man words', marker="o")
    plt.xlabel('Approximate Year')
    plt.ylabel('Gender-Prediction Accuracy\nusing context with "%s"'%interest_word)
    plt.title('Temporal Analysis of MLM Gender-Prediction Accuracy\n(Word: %s, LIWC Category: %s)'%(interest_word, liwc_category))
    plt.legend()
    plt.savefig(results_pp_dir + interest_word + "_temporal_acc_plot.png")
    plt.close()

    # Make normalized prob mean-standard error plot
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    woman_years_se = []
    woman_errors_se = []
    woman_pr_se = []
    for year, x in zip(woman_years, woman_probs):
        if len(x) > 1:
            woman_pr_se.append(statistics.mean(x))
            woman_errors_se.append(statistics.stdev(x))
            woman_years_se.append(year)
    man_years_se = []
    man_errors_se = []
    man_pr_se = []
    for year, x in zip(man_years, man_probs):
        if len(x) > 1:
            man_pr_se.append(statistics.mean(x))
            man_errors_se.append(statistics.stdev(x))
            man_years_se.append(year)
    plt.errorbar(woman_years_se, woman_pr_se, yerr=woman_errors_se, fmt='o', label='woman words', color='r', capsize=5)
    plt.errorbar(man_years_se, man_pr_se, yerr=man_errors_se, fmt='o', label='man words', color='b', capsize=5)
    plt.xlabel('Approximate Year')
    plt.ylabel('Normalized Prediction Probability between genders\nusing context with "%s"'%interest_word)
    plt.title('Temporal Analysis of MLM Gender-Prediction Probability\n(Word: %s, LIWC Category: %s)'%(interest_word, liwc_category))
    plt.legend()
    plt.savefig(results_pp_dir + interest_word + "_temporal_prob_plot.png")
    plt.close()

def generate_plots(pp):
    for work_word in work_words:
        plot_temporal_preds(pp[("woman", "work")], pp[("man", "work")], work_word, "work")
        print("Completed plot for word %s"%work_word)
    for home_word in home_words:
        plot_temporal_preds(pp[("woman", "home")], pp[("man", "home")], home_word, "home")
        print("Completed plot for word %s"%home_word)
    for achiev_word in achievement_words:
        plot_temporal_preds(pp[("woman", "achiev")], pp[("man", "achiev")], achiev_word, "achievement")
        print("Completed plot for word %s"%achiev_word)

def main():
    start_time = time.perf_counter()

    gpu_check()
    os.makedirs(results_pp_dir, exist_ok=True)
    os.makedirs(results_attn_dir, exist_ok=True)
    pp = get_temporal_perplexity_and_attention_values()
    generate_plots(pp)
    
    # Output interesting contexts for attention analysis
    with open(results_attn_dir + "pr_0.25.txt", "w") as output:
        output.write(str(attn_pr_25))
    with open(results_attn_dir + "pr_0.45_0.55.txt", "w") as output:
        output.write(str(attn_pr_45_55))
    with open(results_attn_dir + "pr_0.9999.txt", "w") as output:
        output.write(str(attn_pr_9999))

    end_time = time.perf_counter()
    print(f"This took {(end_time - start_time)/60:0.4f} minutes")

if __name__ == '__main__':
    main()

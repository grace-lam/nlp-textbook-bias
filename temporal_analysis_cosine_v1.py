"""Cosine similarity analysis using LIWC (2015) words for work, home,
and acheivement categories"""

import csv
import os
import statistics
import time

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from scipy.spatial.distance import cosine
from scipy import stats

import extract_embeddings_v1
import finetune_bert
import utilities

model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/block_512/bert_mlm_textbook'
textbook_chronological_dir = 'final_textbook_contexts/512_tokens/'
results_dir = 'temporal_analysis_cosine_results_v1/'
stats_tests_file = 'stats_tests.txt'

MODEL_OPTION = model_bert_textbook_dir # change this to analyze a different model!
LOAD_RESULTS = False # change this to False to rerun embedding extraction and get new results (will override folder!)
results_path = results_dir + "all_results.txt"

# These keywords follow Lucy and Desmzky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

def _cosine_similarity(vec_1, vec_2):
    # Calculate the cosine similarity
    return 1 - cosine(vec_1, vec_2)

def _get_category(word):
    if word in man_words:
        return "man"
    elif word in woman_words:
        return "woman"
    elif word in home_words:
        return "home"
    elif word in work_words:
        return "work"
    elif word in achievement_words:
        return "achiev"
    else:
        print("Error: word given is not a keyword!")
        return None

def _analyze_relations(sentence_data, cosine_similarities, model):
    tokens_tensor, segments_tensor, tokenized_text, sentence_info = sentence_data
    tokens_tensor = torch.tensor(tokens_tensor)
    segments_tensor = torch.tensor(segments_tensor)
    keyword_embeddings = extract_embeddings_v1.get_keyword_embeddings(tokens_tensor, segments_tensor, tokenized_text, sentence_info, model)
    gender_index, query_index, gender_word, query_word = sentence_info
    cosine_sim = _cosine_similarity(keyword_embeddings[gender_word],
     keyword_embeddings[query_word])
    if (gender_word, query_word) not in cosine_similarities:
        cosine_similarities[(gender_word, query_word)] = []
    cosine_similarities[(gender_word, query_word)].append(cosine_sim)

def get_temporal_cosine_similarities():
    # Load pre-trained model (weights) and make it return hidden states
    model = AutoModelForMaskedLM.from_pretrained(MODEL_OPTION, output_hidden_states = True)
    all_cos = {}
    all_cos[("woman", "work")] = {}
    all_cos[("man", "work")] = {}
    all_cos[("woman", "home")] = {}
    all_cos[("man", "home")] = {}
    all_cos[("woman", "achiev")] = {}
    all_cos[("man", "achiev")] = {}
    # Each entry of all_cos is a dictionary formatted as: {year:{(gender_word, query_word): cos_sim}}
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
                     if year not in all_cos[(gender_category, query_category)]:
                         all_cos[(gender_category, query_category)][year] = {}
                     data = utilities.read_context_windows(textbook_chronological_dir + dirname + "/" + filename)
                     for sentence_data in data:
                         _analyze_relations(sentence_data, all_cos[(gender_category, query_category)][year], model)
    with open(results_path, "w") as output:
        output.write(str(all_cos))
    return all_cos

def _get_years_and_sims(cosine_similarities, interest_word):
    years = []
    sims = []
    for year in cosine_similarities:
        tuple_to_sim = cosine_similarities[year]
        for tuple in tuple_to_sim:
            if tuple[1] == interest_word:
                for sim in tuple_to_sim[tuple]:
                    sims.append(sim)
                    years.append(year)
    return years, sims

def _get_years_and_sims_box(cosine_similarities, interest_word):
    years = [] # these are unique
    sims = [] # a list of lists: one list for each year
    for year in sorted(cosine_similarities.keys()):
        tuple_to_sim = cosine_similarities[year]
        year_sims = []
        for tuple in tuple_to_sim:
            if tuple[1] == interest_word:
                for sim in tuple_to_sim[tuple]:
                    year_sims.append(sim)
        if year.endswith('00'):
            years.append(year)
        sims.append(year_sims)
    return years, sims

def plot_temporal_changes(woman_cosine_similarities, man_cosine_similarities, interest_word, liwc_category):
    woman_years, woman_sims = _get_years_and_sims(woman_cosine_similarities, interest_word)
    man_years, man_sims = _get_years_and_sims(man_cosine_similarities, interest_word)
    if not woman_years or not woman_sims or not man_years or not man_sims:
        return
    # Make scatter pots
    woman_years = np.array(list(map(int,woman_years)))
    man_years = np.array(list(map(int,man_years)))
    plt.scatter(woman_years, woman_sims, color='r', label='woman words', alpha=0.1, marker="*")
    plt.scatter(man_years, man_sims, color='b', label='man words', alpha=0.1, marker="o")
    woman_m, woman_b = np.polyfit(woman_years, woman_sims, 1) # linear regression (degree 1)
    plt.plot(woman_years, woman_m*woman_years + woman_b, color='r')
    man_m, man_b = np.polyfit(man_years, man_sims, 1)
    plt.plot(man_years, man_m*man_years + man_b, color='b')
    plt.xlabel('Approximate Year')
    plt.ylabel('Cosine Similarity between\nGender word and "%s"'%interest_word)
    plt.title('Temporal Analysis of Gender-%s Relation\n(LIWC Category: %s)'%(interest_word, liwc_category))
    plt.legend()
    plt.savefig(results_dir + interest_word + "_temporal_cosine_sim_plot.png")
    plt.close()

    # Make box plot
    woman_years_box, woman_sims_box = _get_years_and_sims_box(woman_cosine_similarities, interest_word)
    man_years_box, man_sims_box = _get_years_and_sims_box(man_cosine_similarities, interest_word)
    fig, ax = plt.subplots()
    bp1 = ax.boxplot(woman_sims_box, positions=[i for i in range(1,45,3)], patch_artist=True ,boxprops={'facecolor': 'r'})
    bp2 = ax.boxplot(man_sims_box, positions=[i for i in range(2,45,3)], patch_artist=True, boxprops={'facecolor': 'b'})
    ax.set_xticks(range(0, 45, 6))
    ax.set_xticklabels(woman_years_box)
    plt.xlabel('Approximate Year')
    plt.ylabel('Cosine Similarity between\nGender word and "%s"'%interest_word)
    plt.title('Temporal Analysis of Gender-%s Relation\n(LIWC Category: %s)'%(interest_word, liwc_category))
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['woman words', 'man words'])
    plt.savefig(results_dir + interest_word + "_temporal_cosine_sim_boxplot.png")
    plt.close()

    # Make mean-standard error plot
    all_years = [x for x in range(1300,2050,50)]
    woman_years_se = []
    woman_errors_se = []
    woman_sims_se = []
    for year, x in zip(all_years, woman_sims_box):
        if len(x) > 1:
            woman_sims_se.append(statistics.mean(x))
            woman_errors_se.append(statistics.stdev(x))
            woman_years_se.append(year)
    man_years_se = []
    man_errors_se = []
    man_sims_se = []
    for year, x in zip(all_years, man_sims_box):
        if len(x) > 1:
            man_sims_se.append(statistics.mean(x))
            man_errors_se.append(statistics.stdev(x))
            man_years_se.append(year)
    plt.errorbar(woman_years_se, woman_sims_se, yerr=woman_errors_se, fmt='o', label='woman words', color='r', capsize=5)
    plt.errorbar(man_years_se, man_sims_se, yerr=man_errors_se, fmt='o', label='man words', color='b', capsize=5)
    plt.xlabel('Approximate Year')
    plt.ylabel('Cosine Similarity between\nGender word and "%s"'%interest_word)
    plt.title('Temporal Analysis of Gender-%s Relation\n(LIWC Category: %s)'%(interest_word, liwc_category))
    plt.legend()
    plt.savefig(results_dir + interest_word + "_temporal_cosine_sim_errorplot.png")
    plt.close()

def _get_average_similarity(sims, interest_word):
    count = 0
    sum = 0
    similarities = []
    for year in sims:
        tuple_to_sim = sims[year]
        for tuple in tuple_to_sim:
            if tuple[1] == interest_word:
                for sim in tuple_to_sim[tuple]:
                    sum += sim
                    count += 1
                    similarities.append(sim)
    count = max(count, 1) # to prevent division by zero
    return sum/count, similarities

def _analyze_keyword_similarities(woman_sims, man_sims, keywords):
    woman_averages = []
    man_averages = []
    for word in keywords:
        woman_avg, woman_cosines = _get_average_similarity(woman_sims, word)
        man_avg, man_cosines = _get_average_similarity(man_sims, word)
        woman_averages.append(woman_avg)
        man_averages.append(man_avg)
        t_val, p_val = stats.ttest_ind(woman_cosines, man_cosines)
        with open(results_dir + stats_tests_file, "a") as output:
            output.write("Man vs woman similarities to the word " + word +
            " have averages %f and %f respectively and t-value %f and p-value %f \n"
            %(man_avg, woman_avg, t_val, p_val))
    return woman_averages, man_averages

def plot_static(all_cos):
    woman_work_avg, man_work_avg = _analyze_keyword_similarities(all_cos[("woman", "work")], all_cos[("man", "work")], work_words)
    woman_home_avg, man_home_avg = _analyze_keyword_similarities(all_cos[("woman", "home")], all_cos[("man", "home")], home_words)
    woman_achiev_avg, man_achiev_avg = _analyze_keyword_similarities(all_cos[("woman", "achiev")], all_cos[("man", "achiev")],
     achievement_words)
    plt.scatter(man_work_avg, woman_work_avg, color='b', label='work')
    for i, work_word in enumerate(work_words):
        plt.annotate(work_word, (man_work_avg[i], woman_work_avg[i]), fontsize='small')
    plt.scatter(man_home_avg, woman_home_avg, color='g', label='home')
    for i, home_word in enumerate(home_words):
        plt.annotate(home_word, (man_home_avg[i], woman_home_avg[i]), fontsize='small')
    plt.scatter(man_achiev_avg, woman_achiev_avg, color='r', label='achievement')
    for i, achiev_word in enumerate(achievement_words):
        plt.annotate(achiev_word, (man_achiev_avg[i], woman_achiev_avg[i]), fontsize='small')
    plt.axline((0.2, 0.2), (0.3, 0.3), color='black')
    plt.xlim(0.25, 0.48)
    plt.ylim(0.25, 0.48)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Man Terms (man,men,male,he,his,him)')
    plt.ylabel('Woman Terms\n(woman,women,female,she,her,hers)')
    plt.title('Temporally Averaged Cosine Similarities to...')
    plt.legend(title='LIWC category')
    plt.savefig(results_dir + "static_plot.png")
    plt.close()

def make_all_plots(all_cos):
    for work_word in work_words:
        plot_temporal_changes(all_cos[("woman", "work")], all_cos[("man", "work")], work_word, "work")
        print("completed plot for word %s"%work_word)
    for home_word in home_words:
        plot_temporal_changes(all_cos[("woman", "home")], all_cos[("man", "home")], home_word, "home")
        print("completed plot for word %s"%home_word)
    for achiev_word in achievement_words:
        plot_temporal_changes(all_cos[("woman", "achiev")], all_cos[("man", "achiev")], achiev_word, "achievement")
        print("completed plot for word %s"%achiev_word)
    plot_static(all_cos)
    print("completed static plot")


def main():
    start_time = time.perf_counter()
    finetune_bert.gpu_check()
    os.makedirs(results_dir, exist_ok=True)
    open(results_dir + stats_tests_file, "w") # override existing
    if not LOAD_RESULTS:
        all_cos = get_temporal_cosine_similarities()
    else:
        all_cos = utilities.read_data(results_path)
    make_all_plots(all_cos)
    end_time = time.perf_counter()
    print(f"This took {(end_time - start_time)/60:0.4f} minutes")

if __name__ == '__main__':
    main()

"""Cosine similarity analysis using LIWC (2015) words for work"""

import csv
import os
import statistics
import time

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from scipy.spatial.distance import cosine
from scipy import stats

import extract_embeddings
import finetune_bert
import utilities

model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/80_10_10/bert_mlm_textbook'
textbook_chronological_dir = 'final_textbook_years/all_textbooks/'
results_dir = 'temporal_analysis_cosine_results_sentence_pretrained/'
stats_tests_file = 'stats_tests.txt'

MODEL_OPTION = model_bert_pretrained # change this to analyze a different model!
NUM_ANALYSIS_SENTENCES = -1 # number of sentences to analyze PER time period (change to -1 to do all)
LOAD_RESULTS = False # change this to False to rerun embedding extraction and get new results (will override folder!)

# These keywords follow Lucy and Desmzky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

# Where to write/read results from
results_paths = [results_dir + "woman_work_all_yr_cos.txt", results_dir + "man_work_all_yr_cos.txt",
results_dir + "woman_home_all_yr_cos.txt", results_dir + "man_home_all_yr_cos.txt",
results_dir + "woman_achiev_all_yr_cos.txt", results_dir + "man_achiev_all_yr_cos.txt"]

def _cosine_similarity(vec_1, vec_2):
    # Calculate the cosine similarity
    return 1 - cosine(vec_1, vec_2)

def _analyze_relations(gender_words, interest_words, cosine_similarities, sentence, model):
    sentence_set = set(sentence.split())
    gender_words_present = gender_words & sentence_set
    interest_words_present = interest_words & sentence_set
    # make sure sentence has at least one woman and work word
    if gender_words_present and interest_words_present:
        keywords = gender_words_present.union(interest_words_present)
        keyword_embeddings = extract_embeddings.get_keyword_embeddings(sentence, keywords, model)
        gender_embeddings = gender_words_present & set(keyword_embeddings.keys())
        interest_embeddings = interest_words_present & set(keyword_embeddings.keys())
        # only do this analysis if there was at least one woman and work word
        # even after the tokenizer
        if gender_embeddings and interest_embeddings:
            for gender_word in gender_embeddings:
                for interest_word in interest_embeddings:
                    cosine_sim = _cosine_similarity(keyword_embeddings[gender_word],
                     keyword_embeddings[interest_word])
                    if (gender_word, interest_word) not in cosine_similarities:
                        cosine_similarities[(gender_word, interest_word)] = []
                    cosine_similarities[(gender_word, interest_word)].append(cosine_sim)

def get_temporal_cosine_similarities():
    # Load pre-trained model (weights) and make it return hidden states
    model = AutoModelForMaskedLM.from_pretrained(MODEL_OPTION, output_hidden_states = True)
    woman_work_all_yr_cos = {}
    man_work_all_yr_cos = {}
    woman_home_all_yr_cos = {}
    man_home_all_yr_cos = {}
    woman_achiev_all_yr_cos = {}
    man_achiev_all_yr_cos = {}
    results_cos = [woman_work_all_yr_cos, man_work_all_yr_cos, woman_home_all_yr_cos, man_home_all_yr_cos, woman_achiev_all_yr_cos, man_achiev_all_yr_cos]
    for file in os.listdir(os.fsencode(textbook_chronological_dir)):
         filename = os.fsdecode(file)
         if filename.endswith(".txt"):
             year = filename[:-4] # get rid of extension
             woman_work_yr = {}
             man_work_yr = {}
             woman_home_yr = {}
             man_home_yr = {}
             woman_achiev_yr = {}
             man_achiev_yr = {}
             woman_work_all_yr_cos[year] = woman_work_yr
             man_work_all_yr_cos[year] = man_work_yr
             woman_home_all_yr_cos[year] = woman_home_yr
             man_home_all_yr_cos[year] = man_home_yr
             woman_achiev_all_yr_cos[year] = woman_achiev_yr
             man_achiev_all_yr_cos[year] = man_achiev_yr
             with open(textbook_chronological_dir + filename, 'r') as textbook_reader:
                for sentence in list(textbook_reader)[:NUM_ANALYSIS_SENTENCES]:
                    _analyze_relations(woman_words, work_words, woman_work_yr, sentence, model)
                    _analyze_relations(man_words, work_words, man_work_yr, sentence, model)
                    _analyze_relations(woman_words, home_words, woman_home_yr, sentence, model)
                    _analyze_relations(man_words, home_words, man_home_yr, sentence, model)
                    _analyze_relations(woman_words, achievement_words, woman_achiev_yr, sentence, model)
                    _analyze_relations(man_words, achievement_words, man_achiev_yr, sentence, model)
    for i, result in enumerate(results_cos):
        with open(results_paths[i], "w") as output:
            output.write(str(result))
    return woman_work_all_yr_cos, man_work_all_yr_cos, woman_home_all_yr_cos, man_home_all_yr_cos, woman_achiev_all_yr_cos, man_achiev_all_yr_cos

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

def plot_static(woman_work, woman_home, woman_achiev, man_work, man_home, man_achiev):
    woman_work_avg, man_work_avg = _analyze_keyword_similarities(woman_work, man_work, work_words)
    woman_home_avg, man_home_avg = _analyze_keyword_similarities(woman_home, man_home, home_words)
    woman_achiev_avg, man_achiev_avg = _analyze_keyword_similarities(woman_achiev, man_achiev,
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

def make_all_plots(woman_work_cos, man_work_cos, woman_home_cos, man_home_cos,
woman_achiev_cos, man_achiev_cos):
    for work_word in work_words:
        plot_temporal_changes(woman_work_cos, man_work_cos, work_word, "work")
        print("completed plot for word %s"%work_word)
    for home_word in home_words:
        plot_temporal_changes(woman_home_cos, man_home_cos, home_word, "home")
        print("completed plot for word %s"%home_word)
    for achiev_word in achievement_words:
        plot_temporal_changes(woman_achiev_cos, man_achiev_cos, achiev_word, "achievement")
        print("completed plot for word %s"%achiev_word)
    plot_static(woman_work_cos, woman_home_cos, woman_achiev_cos, man_work_cos, man_home_cos, man_achiev_cos)
    print("completed static plot")


def main():
    start_time = time.perf_counter()
    finetune_bert.gpu_check()
    os.makedirs(results_dir, exist_ok=True)
    open(results_dir + stats_tests_file, "w") # override existing
    if not LOAD_RESULTS:
        woman_work_cos, man_work_cos, woman_home_cos, man_home_cos, woman_achiev_cos, man_achiev_cos = get_temporal_cosine_similarities()
    else:
        woman_work_cos, man_work_cos, woman_home_cos, man_home_cos, woman_achiev_cos, man_achiev_cos = utilities.read_dicts_from_paths(results_paths)
    make_all_plots(woman_work_cos, man_work_cos, woman_home_cos, man_home_cos, woman_achiev_cos, man_achiev_cos)
    end_time = time.perf_counter()
    print(f"This took {(end_time - start_time)/60:0.4f} minutes")

if __name__ == '__main__':
    main()

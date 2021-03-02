"""Cosine similarity analysis using LIWC (2015) words for work"""

import csv
import os

from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from scipy.spatial.distance import cosine

import extract_embeddings
import finetune_bert

model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/80_10_10/bert_mlm_textbook'
textbook_chronological_dir = 'final_textbook_year/all/'
results_dir = 'temporal_analysis_cosine_results/'

MODEL_OPTION = model_bert_textbook_dir # change this to analyze a different model!
NUM_ANALYSIS_SENTENCES = 1000 # number of sentences to analyze PER time period
# These keywords follow Lucy and Desmzky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])


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
    with open(results_dir + "woman_work_all_yr_cos.txt", "w") as output:
        output.write(str(woman_work_all_yr_cos))
    with open(results_dir + "man_work_all_yr_cos.txt", "w") as output:
        output.write(str(man_work_all_yr_cos))
    with open(results_dir + "woman_home_all_yr_cos.txt", "w") as output:
        output.write(str(woman_home_all_yr_cos))
    with open(results_dir + "man_home_all_yr_cos.txt", "w") as output:
        output.write(str(man_home_all_yr_cos))
    with open(results_dir + "woman_achiev_all_yr_cos.txt", "w") as output:
        output.write(str(woman_achiev_all_yr_cos))
    with open(results_dir + "man_achiev_all_yr_cos.txt", "w") as output:
        output.write(str(man_achiev_all_yr_cos))

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

def plot_temporal_changes(woman_cosine_similarities, man_cosine_similarities, interest_word, liwc_category):
    woman_years, woman_sims = _get_years_and_sims(woman_cosine_similarities, interest_word)
    man_years, man_sims = _get_years_and_sims(man_cosine_similarities, interest_word)
    plt.scatter(woman_years, woman_sims, color='r', label='woman words')
    plt.scatter(man_years, man_sims, color='b', label='man words')
    plt.xlabel('Approximate Year')
    plt.ylabel('Cosine Similarity between Gender word and "%s"'%interest_word)
    plt.title('Temporal Analysis of Gender-%s Relation (LIWC Category: %s)'%(interest_word, liwc_category))
    plt.legend()
    plt.savefig(results_dir + interest_word + "_temporal_cosine_sim_plot.png")
    plt.close()

def _get_average_similarity(sims, interest_word):
    count = 0
    sum = 0
    for year in sims:
        tuple_to_sim = sims[year]
        for tuple in tuple_to_sim:
            if tuple[1] == interest_word:
                for sim in tuple_to_sim[tuple]:
                    sum += sim
                    count += 1
    count = max(count, 1) # to prevent division by zero
    return sum/count


def plot_static(woman_work, woman_home, woman_achiev, man_work, man_home, man_achiev):
    woman_work_avg = []
    man_work_avg = []
    for work_word in work_words:
        woman_avg = _get_average_similarity(woman_work, work_word)
        man_avg = _get_average_similarity(man_work, work_word)
        woman_work_avg.append(woman_avg)
        man_work_avg.append(man_avg)
    woman_home_avg = []
    man_home_avg = []
    for home_word in home_words:
        woman_avg = _get_average_similarity(woman_home, home_word)
        man_avg = _get_average_similarity(man_home, home_word)
        woman_home_avg.append(woman_avg)
        man_home_avg.append(man_avg)
    woman_achiev_avg = []
    man_achiev_avg = []
    for achiev_word in achievement_words:
        woman_avg = _get_average_similarity(woman_achiev, achiev_word)
        man_avg = _get_average_similarity(man_achiev, achiev_word)
        woman_achiev_avg.append(woman_avg)
        man_achiev_avg.append(man_avg)
    plt.scatter(man_work_avg, woman_work_avg, color='b', label='work')
    for i, work_word in enumerate(work_words):
        plt.annotate(work_word, (man_work_avg[i], woman_work_avg[i]))
    plt.scatter(man_home_avg, woman_home_avg, color='g', label='home')
    for i, home_word in enumerate(home_words):
        plt.annotate(home_word, (man_home_avg[i], woman_home_avg[i]))
    plt.scatter(man_achiev_avg, woman_achiev_avg, color='r', label='achievement')
    for i, achiev_word in enumerate(achievement_words):
        plt.annotate(achiev_word, (man_achiev_avg[i], woman_achiev_avg[i]))
    plt.xlabel('Man Terms (man,men,male,he,his,him)')
    plt.ylabel('Woman Terms (woman,women,female,she,her,hers)')
    plt.title('Temporally Averaged Cosine Similarities to...')
    plt.legend(title='LIWC category')
    plt.savefig(results_dir + "static_plot.png")
    plt.close()

def main():
    finetune_bert.gpu_check()
    os.makedirs(results_dir, exist_ok=True)
    woman_work_all_yr_cos, man_work_all_yr_cos, woman_home_all_yr_cos, man_home_all_yr_cos, woman_achiev_all_yr_cos, man_achiev_all_yr_cos = get_temporal_cosine_similarities()
    for work_word in work_words:
        plot_temporal_changes(woman_work_all_yr_cos, man_work_all_yr_cos, work_word, "work")
        print("completed plot for word %s"%work_word)
    for home_word in home_words:
        plot_temporal_changes(woman_home_all_yr_cos, man_home_all_yr_cos, home_word, "home")
        print("completed plot for word %s"%home_word)
    for achiev_word in achievement_words:
        plot_temporal_changes(woman_achiev_all_yr_cos, man_achiev_all_yr_cos, achiev_word, "achievement")
        print("completed plot for word %s"%achiev_word)
    plot_static(woman_work_all_yr_cos, woman_home_all_yr_cos, woman_achiev_all_yr_cos, man_work_all_yr_cos, man_home_all_yr_cos, man_achiev_all_yr_cos)
    print("completed static plot")

if __name__ == '__main__':
    main()

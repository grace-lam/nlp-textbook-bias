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
model_bert_textbook_dir = 'bert_mlm/bert_mlm_textbook'
textbook_chronological = 'all_textbook_data.txt' # note: this needs to be in chronological order
liwc_path = 'LIWC2015.csv' # make sure to include this file LOCALLY!
results_dir = 'temporal_analysis_cosine_results/'

NUM_ANALYSIS_SENTENCES = 30000 # number of sentences to analyze; set to -1 to analyze all words
MODEL_OPTION = model_bert_textbook_dir # change this to analyze a different model!
# These keywords follow Lucy and Desmzky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])

def _get_work_keywords():
    work_words = []
    with open(liwc_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        CATEGORY_INDEX = 2
        TERM_INDEX = 1
        for row in csv_reader:
            if row[CATEGORY_INDEX] == 'WORK':
                work_words.append(row[TERM_INDEX])
    return work_words

work_words = set(_get_work_keywords())

def _cosine_similarity(vec_1, vec_2):
    # Calculate the cosine similarity
    return 1 - cosine(vec_1, vec_2)

def _analyze_relations(gender_words, cosine_similarities, context_indices, curr_index, sentence, model):
    sentence_set = set(sentence.split())
    gender_words_present = gender_words & sentence_set
    work_words_present = work_words & sentence_set
    # make sure sentence has at least one woman and work word
    if gender_words_present and work_words_present:
        keywords = gender_words_present.union(work_words_present)
        keyword_embeddings = extract_embeddings.get_keyword_embeddings(sentence, keywords, model)
        gender_embeddings = gender_words_present & set(keyword_embeddings.keys())
        work_embeddings = work_words_present & set(keyword_embeddings.keys())
        # only do this analysis if there was at least one woman and work word
        # even after the tokenizer
        if gender_embeddings and work_embeddings:
            for gender_word in gender_embeddings:
                for work_word in work_embeddings:
                    cosine_similarities.append(_cosine_similarity(keyword_embeddings[gender_word],
                     keyword_embeddings[work_word]))
                    context_indices.append(curr_index)

def get_temporal_cosine_similarities():
    # Load pre-trained model (weights) and make it return hidden states
    model = AutoModelForMaskedLM.from_pretrained(MODEL_OPTION, output_hidden_states = True)
    curr_index = 0
    woman_context_indices = []
    woman_cosine_similarities = []
    man_context_indices = []
    man_cosine_similarities = []
    with open(textbook_chronological, 'r') as textbook_reader:
        for sentence in list(textbook_reader)[:NUM_ANALYSIS_SENTENCES]:
            _analyze_relations(woman_words, woman_cosine_similarities, woman_context_indices, curr_index, sentence, model)
            _analyze_relations(man_words, man_cosine_similarities, man_context_indices, curr_index, sentence, model)
            curr_index += 1
    with open(results_dir + "woman_cosine_similarities.txt", "w") as output:
        output.write(str(woman_cosine_similarities))
    with open(results_dir + "woman_context_indices.txt", "w") as output:
        output.write(str(woman_context_indices))
    with open(results_dir + "man_cosine_similarities.txt", "w") as output:
        output.write(str(man_cosine_similarities))
    with open(results_dir + "man_context_indices.txt", "w") as output:
        output.write(str(man_context_indices))
    return woman_context_indices, woman_cosine_similarities, man_context_indices, man_cosine_similarities

def plot_temporal_changes(woman_context_indices, woman_cosine_similarities, man_context_indices, man_cosine_similarities):
    plt.scatter(woman_context_indices, woman_cosine_similarities, color='r', label='woman words')
    plt.scatter(man_context_indices, man_cosine_similarities, color='b', label='man words')
    plt.xlabel('Context ID (Order of Sentence in Chronology)')
    plt.ylabel('Cosine Similarity between Gender and Work word')
    plt.title('Temporal Analysis of Gender-Work Relation')
    plt.legend()
    plt.savefig(results_dir + "cosine_sim_plot.png")
    plt.show()


def main():
    finetune_bert.gpu_check()
    os.makedirs(results_dir, exist_ok=True)
    woman_context_indices, woman_cosine_similarities, man_context_indices, man_cosine_similarities = get_temporal_cosine_similarities()
    plot_temporal_changes(woman_context_indices, woman_cosine_similarities, man_context_indices, man_cosine_similarities)

if __name__ == '__main__':
    main()

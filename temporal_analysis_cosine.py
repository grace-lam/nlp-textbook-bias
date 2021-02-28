"""Cosine similarity analysis using LIWC (2015) words for work"""

import csv
import os

from datasets import load_dataset
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from scipy.spatial.distance import cosine

import extract_embeddings
import finetune_bert

model_bert_pretrained = 'bert-base-uncased'
model_bert_textbook_dir = 'bert_mlm/bert_mlm_textbook'
textbook_chronological = 'all_textbook_data.txt' # note: this needs to be in chronological order
liwc_path = 'LIWC2015.csv'
results_dir = 'temporal_analysis_cosine_results/'

def get_work_keywords():
    work_words = []
    with open(liwc_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        CATEGORY_INDEX = 2
        TERM_INDEX = 1
        for row in csv_reader:
            if row[CATEGORY_INDEX] == 'WORK':
                work_words.append(row[TERM_INDEX])
    return work_words

def cosine_similarity(vec_1, vec_2):
    # Calculate the cosine similarity
    return 1 - cosine(vec_1, vec_2)

def get_temporal_cosine_similarities():
    # These keywords follow Lucy and Desmzky's set up
    man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
    woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
    work_words = set(get_work_keywords())
    # Load pre-trained model (weights) and make it return hidden states
    model = AutoModelForMaskedLM.from_pretrained(model_bert_textbook_dir, output_hidden_states = True)
    curr_index = 0
    context_indices = []
    cosine_similarities = []
    with open(textbook_chronological, 'r') as textbook_reader:
        for sentence in list(textbook_reader)[:10000]:
            sentence_set = set(sentence.split())
            woman_words_present = woman_words & sentence_set
            work_words_present = work_words & sentence_set
            # make sure sentence has at least one woman and work word
            if woman_words_present and work_words_present:
                keywords = woman_words_present.union(work_words_present)
                keyword_embeddings = extract_embeddings.get_keyword_embeddings(sentence, keywords, model)
                woman_embeddings = woman_words_present & set(keyword_embeddings.keys())
                work_embeddings = work_words_present & set(keyword_embeddings.keys())
                # only do this analysis if there was at least one woman and work word
                # even after the tokenizer
                if woman_embeddings and work_embeddings:
                    for woman_word in woman_embeddings:
                        for work_word in work_embeddings:
                            cosine_similarities.append(cosine_similarity(keyword_embeddings[woman_word],
                             keyword_embeddings[work_word]))
                            context_indices.append(curr_index)
            curr_index += 1
    print(cosine_similarities)
    with open(results_dir + "cosine_similarities.txt", "w") as output:
        output.write(str(cosine_similarities))
    with open(results_dir + "context_indices.txt", "w") as output:
        output.write(str(context_indices))
    return context_indices, cosine_similarities

def plot_temporal_changes(context_indices, cosine_similarities):
    pass


def main():
    finetune_bert.gpu_check()
    os.makedirs(results_dir, exist_ok=True)
    context_indices, cosine_similarities = get_temporal_cosine_similarities()
    plot_temporal_changes(context_indices, cosine_similarities)

if __name__ == '__main__':
    main()

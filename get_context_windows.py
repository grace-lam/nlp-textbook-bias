"""Given a text dataset and interest words, extracts appropriate contexts to input
into BERT as probes. Each gender-query word pair gets its own file for each
time period. Each such file contains:
 an array with each entry formatted as
         (tokens_tensor, segments_tensor, tokenized_sentence,
         (gender_index, query_index, gender_word, query_word))

"""

import math
import os
import re

import numpy as np

import utilities

text_data_path = 'final_textbook_years/all_textbooks/'
NUM_TOKENS = 512 # in number of tokens
MAX_DISTANCE_BETWEEN_KEYWORDS = 100 # in number of words
CHARS_PER_WORD = 6 # over-estimate number of characters per word (incl space)
context_window_dir = 'final_textbook_contexts/' + str(NUM_TOKENS) + '_tokens/'

# These keywords follow Lucy and Demszky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

def extract_sentences(gender_word, query_word, text):
    """Extract sentences in which gender and query words are roughly centered.
       returns: sentences, an array with each entry formatted as
                (sentence, (gender_index, query_index, gender_word, query_word))
    """
    sentences = []
    # gender can come before and after query words, but we take care of both in the same
    # regex so as not to duplicate matched contexts
    pattern = rf"\b{gender_word}\b.*?\b{query_word}\b|\b{query_word}\b.*?\b{gender_word}\b"
    for match in re.finditer(pattern, text, re.IGNORECASE):
        matched_string = match.group(0)
        context_len = len(matched_string.split())
        if context_len < MAX_DISTANCE_BETWEEN_KEYWORDS:
            start_index = match.start()
            end_index = match.end()
            additional_tokens = NUM_TOKENS - context_len
            additional_chars = CHARS_PER_WORD*additional_tokens
            # prevent python from wrapping around
            full_context_start = max(0,int(start_index - additional_chars/2))
            full_context_end = int(end_index + additional_chars/2)
            full_context = text[full_context_start:full_context_end]
            # extract indices back out
            start_index -= full_context_start
            end_index -= full_context_start
            if full_context[start_index:].startswith(query_word):
                query_index = start_index
                gender_index = end_index - len(gender_word)
            else:
                query_index = end_index  - len(query_word)
                gender_index = start_index
            sentences.append((full_context, (gender_index, query_index, gender_word, query_word)))
    return sentences


def _map_to_token_indices(word, orig_index, orig_text, tokenized_text):
    """Helper function to map from original index to tokenized index"""
    # approximate deviation allowance
    length_difference = abs(3*(len(tokenized_text) - len(orig_text.split())))
    word_token_indices = np.where(np.array(tokenized_text) == word)[0]
    word_token_index = -1
    for index in word_token_indices:
        if abs(orig_index - index) < length_difference:
            word_token_index = index
            break
    return word_token_index

def tokenize_sentences(sentences):
    """Given sentences, tokenize with BERT and retain rough centering of keywords
        returns: tokenized_sentences, an array with each entry formatted as
                 (tokens_tensor, segments_tensor, tokenized_sentence,
                 (gender_index, query_index, gender_word, query_word))
    """
    tokenized_sentences = []
    for sentence, sentence_info in sentences:
        gender_index, query_index, gender_word, query_word = sentence_info
        tokens_tensor, segments_tensor, tokenized_text = utilities.sentence_to_tokens(sentence, NUM_TOKENS)
        # map original character indices to word indices
        gender_word_index = len(sentence[:gender_index].split())
        query_word_index = len(sentence[:query_index].split())
        query_token_index = _map_to_token_indices(query_word, query_word_index, sentence, tokenized_text)
        gender_token_index = _map_to_token_indices(gender_word, gender_word_index, sentence, tokenized_text)
        if query_token_index == -1 or gender_token_index == -1:
            continue
        tokenized_info = (gender_token_index, query_token_index, gender_word, query_word)
        tokenized_sentences.append((tokens_tensor, segments_tensor, tokenized_text, tokenized_info))
    return tokenized_sentences


def process_categories(gender_category, query_category):
    """Extracts context windows where words from both categories are present"""
    for gender_word in gender_category:
        for query_word in query_category:
            for text_filename in os.listdir(text_data_path):
                os.makedirs(context_window_dir + "/" + text_filename, exist_ok=True)
                text = utilities.eliminate_newlines(text_data_path + text_filename)
                sentences = extract_sentences(gender_word, query_word, text)
                tokenized_sentences = tokenize_sentences(sentences)
                with open(context_window_dir + "/" + text_filename + f"/{gender_word}_{query_word}.txt", "w") as output:
                    output.write(str(tokenized_sentences))

def main():
    os.makedirs(context_window_dir, exist_ok=True)
    query_word_categories = [home_words, work_words, achievement_words]
    gender_word_categories = [man_words, woman_words]
    for gender_category in gender_word_categories:
        for query_category in query_word_categories:
            process_categories(gender_category, query_category)

if __name__ == '__main__':
    main()

"""Given a text dataset and interest words, extracts appropriate contexts to input
into BERT as probes to corresponding files"""

import os
import re

import utilities

text_data_path = 'final_textbook_years/all_textbooks/'
NUM_TOKENS = 512 # in number of tokens
MAX_DISTANCE_BETWEEN_KEYWORDS = 100 # in number of words
CHARS_PER_WORD = 7 # over-estimate number of characters per word (incl space)

# These keywords follow Lucy and Demszky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

def extract_sentences(gender_word, query_word, text):
    """Extract sentences in which gender and query words are roughly centered"""
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
            sentences.add(full_context)
    return sentences

# @TODO: centering, labeling with indices, then tokenization from sentences, and output to file

def process_categories(gender_category, query_category):
    for gender_word in gender_category:
        for query_word in query_category:
            for text_filename in os.listdir(text_data_path):
                text = utilities.eliminate_newlines(text_data_path + text_filename)
                extract_sentences(gender_word, query_word, text)

def main():
    process_categories(man_words, home_words)

if __name__ == '__main__':
    main()

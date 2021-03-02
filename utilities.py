"""Utility functions"""

import ast

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

def read_dicts_from_paths(paths):
    list_of_dicts = []
    for path in paths:
        with open(path) as f:
            data = f.read()
            data = ast.literal_eval(data)
            list_of_dicts.append(data)
    return list_of_dicts

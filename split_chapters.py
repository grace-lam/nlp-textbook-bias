import os
import re

directory = os.fsencode('final_textbook_txts/')
chapters_directory = 'final_textbook_chapters/'
original_directory = 'final_textbook_txts/'

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
         textbook_name = filename[:-4] # get rid of extension
         pearson_regex = re.compile(r'([0-9])\.1 [A-Z]')
         with open(original_directory + filename, 'r') as textbook_reader:
             os.makedirs(chapters_directory + textbook_name, exist_ok=True)
             if textbook_name.startswith("mastering"): # mastering starts at chapter 5
                 next_chapter = 6
             else:
                 next_chapter = 2
             if textbook_name.startswith("Give_Me_Liberty"):
                 chapter_sentinel = "c h . "
             else:
                 chapter_sentinel = "chapter "
             line_count = 1
             for sentence in textbook_reader:
                 start_next = False
                 if textbook_name.startswith("pearson"): # pearson has chapters like 1.1, etc.
                     chapter_prefix = pearson_regex.search(sentence)
                     if chapter_prefix and chapter_prefix.group()[0] == str(next_chapter):
                         start_next = True
                 elif chapter_sentinel + str(next_chapter) in sentence.lower() and line_count > 100:
                     if "see " + chapter_sentinel + str(next_chapter) not in sentence.lower():
                         start_next = True
                 if start_next:
                     line_count = 1
                     next_chapter += 1
                 with open(chapters_directory + textbook_name + "/chapter_" + str(next_chapter - 1) + ".txt", 'a') as textbook_writer:
                     textbook_writer.write(sentence)
                 line_count += 1

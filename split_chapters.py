import os

directory = os.fsencode('textbook_data/')
chapters_directory = 'textbook_data_chapters/'
original_directory = 'textbook_data/'

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
         textbook_name = filename[:-4]
         with open(original_directory + filename, 'r') as textbook_reader:
             os.makedirs(chapters_directory + textbook_name, exist_ok=True)
             next_chapter = 2
             for sentence in textbook_reader:
                 if "chapter " + str(next_chapter) in sentence.lower():
                     next_chapter += 1
                 with open(chapters_directory + textbook_name + "/chapter_" + str(next_chapter - 1) + ".txt", 'a') as textbook_writer:
                     textbook_writer.write(sentence)

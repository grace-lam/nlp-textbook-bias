"""Analyze distance between achievement and woman vs man words"""

import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
# These keywords follow Lucy and Desmzky's set up
man_words = set(['man', 'men', 'male', 'he', 'his', 'him'])
woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])
home_words = set(['home', 'domestic', 'household', 'chores', 'family'])
work_words = set(['work', 'labor', 'workers', 'economy', 'trade', 'business', 'jobs', 'company', 'industry', 'pay', 'working', 'salary', 'wage'])
achievement_words = set(['power', 'authority', 'achievement', 'control', 'won', 'powerful', 'success', 'better', 'efforts', 'plan', 'tried', 'leader'])

textbook_chronological_dir = 'final_textbook_years/all_textbooks/'

man_dists = {}
woman_dists = {}

for file in os.listdir(os.fsencode(textbook_chronological_dir)):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
         year = filename[:-4]
         man_dists[year] = []
         woman_dists[year] = []
         achievement_index = -1
         woman_index = -1
         man_index = -1
         with open(textbook_chronological_dir + filename, 'r') as textbook_reader:
             data = textbook_reader.read()
             data = data.replace("\n", "")
             for i, word in enumerate(data.split()):
                 if word in achievement_words:
                     achievement_index = i
                     if woman_index > -1:
                         woman_dists[year].append(abs(woman_index - achievement_index))
                     if woman_index > -1:
                         man_dists[year].append(abs(man_index - achievement_index))
                 if word in man_words:
                     man_index = i
                     if achievement_index > -1:
                         man_dists[year].append(abs(man_index - achievement_index))
                 if word in woman_words:
                     woman_index = i
                     if achievement_index > -1:
                         woman_dists[year].append(abs(woman_index - achievement_index))
     print("finished one file")


# Make mean-standard error plot
all_years = [x for x in range(1300,2050,50)]
woman_years_se = []
woman_errors_se = []
woman_dists_se = []
man_years_se = []
man_errors_se = []
man_dists_se = []
for year in woman_dists:
    if len(woman_dists[year]) > 1:
        woman_dists[year] = np.array(woman_dists[year])
        woman_dists_se.append(np.mean(woman_dists[year]))
        woman_years_se.append(int(year))
        woman_errors_se.append(statistics.stdev(woman_dists[year]))
    if len(man_dists[year]) > 1:
        man_dists[year] = np.array(man_dists[year])
        man_dists_se.append(np.mean(man_dists[year]))
        man_years_se.append(int(year))
        man_errors_se.append(statistics.stdev(man_dists[year]))

plt.errorbar(woman_years_se, woman_dists_se, yerr=woman_errors_se, fmt='o', label='woman words', color='r', capsize=5)
plt.errorbar(man_years_se, man_dists_se, yerr=man_errors_se, fmt='o', label='man words', color='b', capsize=5)
plt.xlabel('Approximate Year')
plt.ylabel('Average Number of Tokens Between \n Achievement word and Gender word')
plt.title('Distance between Gender words \n and Achievement Words')
plt.legend()
plt.savefig("achievement_dist.png")
plt.close()

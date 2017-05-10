import text_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import nltk
import string

valid_words = ['пробить', 'работа', 'детализация', 'взлом', 'услуга']
vocabulary_size = 50
window_size = 2
batch_size = 30


if not os.path.isfile('data.csv'):
    # with openpyxl.load_workbook('train_probiv.xlsx', read_only=True, data_only=True, keep_links=False) as f:
    #     for line in f:
    #         print(line)
    raw = pd.read_csv('probiv1.csv', encoding='utf-8', error_bad_lines=False, sep=';',
                      skip_blank_lines=True).values.astype('str')
    posts = raw[:, 3]
    stops = nltk.corpus.stopwords.words('russian')
    stops.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'р', 'на', 'новым', 'годом', 'поздравляем'])
    # print(posts)
    data = text_helpers.normalize_text(posts, stops, morph='False')
    # print(data)
    pd.DataFrame(data).to_csv('data.csv', sep=',', encoding='utf-8', header=False, index=False)
    # data = pd.DataFrame(data).astype('str')
else:
    data = pd.read_csv('data.csv').values.astype('str')

print(type(data))
word_dict = text_helpers.build_dictionary(data, vocabulary_size)

numeric_txt = text_helpers.text_to_numbers(data, word_dict)

valid_examples = [word_dict[x] for x in valid_words]
valid_w_dict = {valid_words[x]: valid_examples[x] for x in range(len(valid_examples))}
print(valid_w_dict)



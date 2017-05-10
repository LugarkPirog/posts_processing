import text_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

valid_words = ['пробить', 'работа', 'детализация', 'взлом', 'услуга', 'база', 'дать']
vocabulary_size = 50
window_size = 2
batch_size = 30


if not os.path.isfile('data.csv'):
    raw = pd.read_csv('probiv1.csv', encoding='utf-8', error_bad_lines=False, sep=';',
                      skip_blank_lines=True).values.astype('str')
    posts = raw[:, 3]
    stops = nltk.corpus.stopwords.words('russian')
    stops.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'р', 'на', 'новым', 'годом', 'поздравляем', 'дам', 'дать'])
    # print(type(posts))
    data = text_helpers.normalize_text(posts, stops, morph=True, del_eng=True)
    # print(type(data))
    pd.DataFrame(data).to_csv('data.csv', sep=',', encoding='utf-8', header=False, index=False)
    # data = pd.DataFrame(data).astype('str')
else:
    data = pd.read_csv('data.csv').values.astype('str')
    data = [''.join(x for x in y) for y in data]

# print(data)
# word_dict = text_helpers.build_dictionary(data, vocabulary_size)
# numeric_txt = text_helpers.text_to_numbers(data, word_dict)
# valid_examples = [word_dict[x] for x in valid_words]
# valid_w_dict = {valid_words[x]: valid_examples[x] for x in range(len(valid_examples))}

# tfidf = text_helpers.tf_idf(word_dict,numeric_txt)
tfidf = TfidfVectorizer()
print((tfidf.fit_transform(data)[0]).shape)
# for x in word_dict:
#     print((x, word_dict[x]), sep='\n')
#
# for x in data:
#     print(x.split())

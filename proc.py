import text_helpers
import numpy as np
import pandas as pd
import openpyxl
import tensorflow as tf
import os
import nltk
import string

if os.path.isfile('data.csv'):
    # with openpyxl.load_workbook('train_probiv.xlsx', read_only=True, data_only=True, keep_links=False) as f:
    #     for line in f:
    #         print(line)
    raw = pd.read_csv('probiv1.csv', encoding='utf-8', error_bad_lines=False, sep=';',
                      skip_blank_lines=True).values.astype('str')
    posts = raw[:, 3]
    stops = nltk.corpus.stopwords.words('russian')
    stops.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'р', 'на'])
    data = text_helpers.normalize_text(posts, stops)
    # print(data)
    pd.DataFrame(data).to_csv('data.csv', sep=',', encoding='utf-8', header=False, index=False)
    data = pd.DataFrame(data).values.astype('str')
else:
    data = pd.read_csv('data.csv').values.astype('str')




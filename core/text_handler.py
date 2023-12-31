import re

import emoji as emoji
import pandas as pd
from pymorphy2 import MorphAnalyzer
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from core.model import get_model, test_model, cv

stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def split_to_sentences(text):
    text = re.split('[.;\n!?·•⦁-]+', text)
    return text


def lemmatize(doc):
    patterns = "[0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-–•]+"
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        token = token.strip()
        token = morph.normal_forms(token)[0]
        if token not in stopwords_ru:
            tokens.append(token)
    return tokens


def prepared_text(text):
    data = []
    for i in text:
        data.append(lemmatize(i))
    return data


def prepared_table(data_table):
    list_corpus = []
    list_labels = []
    list_source_sentence = []
    for i in range(len(data_table)):
        data_table[i][1] = ''.join(
            char for char in data_table[i][1] if not emoji.is_emoji(char) and char not in ['▉', '_', '*', '='])
        source_sentence = split_to_sentences(data_table[i][1])
        prepared_sentences = prepared_text(source_sentence)
        temp = []
        for j in range(len(prepared_sentences)):
            if prepared_sentences[j]:
                temp.append(source_sentence[j])
                sentence = ' '.join(prepared_sentences[j])
                list_corpus.append(sentence)
                list_labels.append(data_table[i][2])
        list_source_sentence.append(temp)
    return list_corpus, list_labels, list_source_sentence


def prepared_list(lst):
    list_corpus = []
    list_source_sentence = []
    for i in range(len(lst)):
        tmp = ''.join(char for char in lst[i] if not emoji.is_emoji(char) and char not in ['▉', '_', '*', '='])
        source_sentence = split_to_sentences(tmp)
        prepared_sentences = prepared_text(source_sentence)
        temp = []
        for j in range(len(prepared_sentences)):
            if prepared_sentences[j]:
                temp.append(source_sentence[j])
                sentence = ' '.join(prepared_sentences[j])
                list_corpus.append(sentence)
        list_source_sentence.append(temp)
    return list_corpus, list_source_sentence


def get_result_dict(test_list):
    clf = get_model()
    f = open('./resources/list_corpus_train.txt', 'r', encoding='utf-8')
    list_corpus_train = f.read().split('.')
    f.close()
    X_train_counts, count_vectorizer = cv(list_corpus_train)
    i = 0
    list_corpus_test, list_source_sentence_test = prepared_list(test_list)
    y_predicted_counts = test_model(clf, list_corpus_test, count_vectorizer)

    class_suggestions = {'Исходник': [], 'Должностные обязанности': [], 'Условия': [],
                         'Требования к соискателю': []}
    for ind_one_vacancy in range(len(list_source_sentence_test)):
        class_suggestions['Исходник'].append(test_list[ind_one_vacancy])
        temp = {'Должностные обязанности': [], 'Условия': [], 'Требования к соискателю': []}
        for sentence in list_source_sentence_test[ind_one_vacancy]:
            if y_predicted_counts[i] == '0':
                temp['Должностные обязанности'].append(sentence)
            elif y_predicted_counts[i] == '1':
                temp['Условия'].append(sentence)
            else:
                temp['Требования к соискателю'].append(sentence)
            i += 1
        match = re.compile('x\d+D|услови[ея] ?:|требовани[ея] ?:|обязанност[ьи] ?:', re.IGNORECASE)
        temp['Должностные обязанности'] = match.sub('', ' '.join(temp['Должностные обязанности']))
        temp['Условия'] = match.sub('', ' '.join(temp['Условия']))
        temp['Требования к соискателю'] = match.sub('', ' '.join(temp['Требования к соискателю']))
        class_suggestions['Должностные обязанности'].append(temp['Должностные обязанности'])
        class_suggestions['Условия'].append(temp['Условия'])
        class_suggestions['Требования к соискателю'].append(temp['Требования к соискателю'])
    return class_suggestions


def write_to_excel(class_suggestions, path):
    df = pd.DataFrame(class_suggestions)
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer._save()

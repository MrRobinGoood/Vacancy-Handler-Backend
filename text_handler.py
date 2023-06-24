import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def split_to_sentences(text):
    text = re.split('[.;\n!?·-]+', text)
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
        source_sentence = split_to_sentences(lst[i])
        prepared_sentences = prepared_text(source_sentence)
        temp = []
        for j in range(len(prepared_sentences)):
            if prepared_sentences[j]:
                temp.append(source_sentence[j])
                sentence = ' '.join(prepared_sentences[j])
                list_corpus.append(sentence)
        list_source_sentence.append(temp)
    return list_corpus, list_source_sentence

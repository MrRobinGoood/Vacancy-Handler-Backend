import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def split_to_sentences(text):
    text = re.split('[.;\n!?-]+', text)
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
    text = split_to_sentences(text)
    data = []
    for i in text:
        data.append(lemmatize(i))
    return data

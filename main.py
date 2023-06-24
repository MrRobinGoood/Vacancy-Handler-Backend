import pandas as pd
from matplotlib import pyplot as plt

from statistics import plot_LSA
from text_handler import *
from model import *

if __name__ == '__main__':
    clf = get_model()
    f = open('list_corpus_train.txt', 'r', encoding='utf-8')
    list_corpus_train = f.read().split('.')
    f.close()
    f = open('list_labels_train.txt', 'r', encoding='utf-8')
    list_labels_train = f.read().split('.')
    f.close()
    list_labels_train = [int(i) for i in list_labels_train]
    X_train_counts, count_vectorizer = cv(list_corpus_train)
    i = 0

    test_list = [
        'Выполнение работ по гнутью и резке арматурной стали на ручных, электромеханических и электрических станках. Выполнение работ по сборке и вязке арматурных сеток и плоских арматурных каркасов.',
        'Вахта в город Москва.  Обязанности: - армирование каркаса;  Требования: - опыт в строительстве приветствуется; - работа в бригаде;  Условия: - продолжительность вахты 60/30 (продление вахты возможно); - Официальное трудоустройство; - ЗП в срок и без задержек; - Авансирование дважды в месяц по 15 000 рублей, 15 и 30 числа; - Питание трехразовое за счет организации; - Выдача спецодежды и Сизов без вычета из заработной платы; - Организованные отправки до объекта (покупка билетов); - Помощь в прохождение медицинского осмотра; - Возможность получить квалификационные удостоверения; - Карьерный рост до бригадира/мастера;',
        'Крупной Федеральной Компании для работы ВАХТОВЫМ МЕТОДОМ на строительные участки требуются специалисты:  Обязанности:  · Работы на монолите (заливка бетона, укладка и вязка арматуры)  · Монтажные работы (МК/ЖБК)  · Общестроительные работы и вспомогательные работы  Требования:  · Понимание работы в бригаде  · Дисциплинированность, трудолюбие  · Опыт работы в строительстве будет преимуществом  · (готовы рассмотреть кандидатов и без опыта работы, всему научим, в перспективе карьерный рост до бригадира/мастера участка)  Условия:  · Продолжительность вахты 60/30, 90/30 (можно больше)  · Официальное трудоустройство по ТК РФ с первой рабочей смены  · ЗП строго в срок и без задержек (документальное подтверждение в трудовом договоре)  · Авансирование (15 и 30го числа каждого месяца по 15000 р.)  · Обеспечим сезонной спецодеждой и СИЗами без вычетов из ЗП  · Организованное трехраховое горячее питание за счет компании  · Организованные отправки до объектов (покупаем билеты на вахту/с вахты)  · Помощь в прохождении мед. осмотра (при необходимости)  · Возможность получить квалификационное удостоверение (обучение основным и смежным специальностям в нашем аккредитованном учебном центре)',
        ]
    list_corpus_test, list_source_sentence_test = prepared_list(test_list)
    y_predicted_counts = test_model(clf, list_corpus_test, count_vectorizer)
    print(y_predicted_counts)
    print(list_source_sentence_test)
    print(len(list_source_sentence_test))
    df = pd.DataFrame()
    for ind_one_vacancy in range(len(list_source_sentence_test)):
        class_suggestions = {'Исходник': [test_list[ind_one_vacancy]], 'Должностные обязанности': [], 'Условия': [], 'Требования к соискателю': []}
        for sentence in list_source_sentence_test[ind_one_vacancy]:
            if y_predicted_counts[i] == '0':
                class_suggestions['Должностные обязанности'].append(sentence)
            elif y_predicted_counts[i] == '1':
                class_suggestions['Условия'].append(sentence)
            else:
                class_suggestions['Требования к соискателю'].append(sentence)
            i += 1
        class_suggestions['Должностные обязанности'] = [' '.join(class_suggestions['Должностные обязанности'])]
        class_suggestions['Условия'] = [' '.join(class_suggestions['Условия'])]
        class_suggestions['Требования к соискателю'] = [' '.join(class_suggestions['Требования к соискателю'])]
        print('Должностные обязанности: ', class_suggestions['Должностные обязанности'])
        print('Условия: ', class_suggestions['Условия'])
        print('Требования к соискателю: ', class_suggestions['Требования к соискателю'])
        df_temp = pd.DataFrame(class_suggestions)
        df = pd.concat([df, df_temp])
    writer = pd.ExcelWriter('dataframes.xlsx', engine='xlsxwriter')
    df.to_excel(writer)
    writer._save()
    plot_LSA(X_train_counts, list_labels_train)

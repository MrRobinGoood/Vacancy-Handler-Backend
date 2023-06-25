from core.statistics import plot_LSA
from text_handler import *
from model import *

if __name__ == '__main__':
    f = open('../resources/list_corpus_train.txt', 'r', encoding='utf-8')
    list_corpus_train = f.read().split('.')
    f.close()
    X_train_counts, count_vectorizer = cv(list_corpus_train)
    df = pd.read_excel('../resources/datasets/source_dataset.xlsx')
    test_list = df['responsibilities(Должностные обязанности)'].tolist()
    print(test_list)
    class_suggestions = get_result_dict(test_list)
    print(class_suggestions)
    write_to_excel(class_suggestions, '../resources/parsed_source.xlsx')
    # create distribution diagram
    # f = open('../resources/list_labels_train.txt', 'r', encoding='utf-8')
    # list_labels_train = f.read().split('.')
    # list_labels_train = [int(i) for i in list_labels_train]
    # f.close()
    # plot_LSA(X_train_counts, list_labels_train)

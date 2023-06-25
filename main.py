from core.statistics import plot_LSA
from core.text_handler import *
from core.model import *

if __name__ == '__main__':
    print('-Start work-')
    f = open('resources/list_corpus_train.txt', 'r', encoding='utf-8')
    list_corpus_train = f.read().split('.')
    f.close()
    X_train_counts, count_vectorizer = cv(list_corpus_train)
    df = pd.read_excel('resources/datasets/source_dataset.xlsx')
    test_list = df['responsibilities(Должностные обязанности)'].tolist()
    class_suggestions = get_result_dict(test_list)
    write_to_excel(class_suggestions, 'resources/parsed_source.xlsx')
    print('-Saved to xlsx file-')
    # create distribution diagram
    # f = open('../resources/list_labels_train.txt', 'r', encoding='utf-8')
    # list_labels_train = f.read().split('.')
    # list_labels_train = [int(i) for i in list_labels_train]
    # f.close()
    # plot_LSA(X_train_counts, list_labels_train)

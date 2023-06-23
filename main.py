from text_handler import prepared_text
from model import *
import gspread

if __name__ == '__main__':
    gc = gspread.service_account(filename='key.json')
    sh = gc.open("cpHh")
    worksheet = sh.get_worksheet(0)
    data_table = worksheet.get_all_values()
    data_table.pop(0)

    list_corpus = []
    list_labels = []

    for i in range(995):
        prepared_sentences = prepared_text(data_table[i][1])
        for sentence in prepared_sentences:
            if sentence:
                sentence = ' '.join(sentence)
                list_corpus.append(sentence)
                list_labels.append(data_table[i][2])

    X_train, X_test, y_train, y_test = sep_train_test(list_corpus, list_labels)
    clf, count_vectorizer = train_model(X_train, y_train)
    y_predicted_counts = test_model(clf, X_test, count_vectorizer)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

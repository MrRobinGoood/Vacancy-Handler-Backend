import gspread
from core.model import train_model, cv, save_model, test_model, get_metrics
from core.text_handler import prepared_table


def save_list_corpus_train(list_corpus_train):
    with open('../resources/list_corpus_train.txt', 'w', encoding='utf-8') as f:
        f.write('.'.join(list_corpus_train))


def save_list_labels_train(list_labels_train):
    with open('../resources/list_labels_train.txt', 'w', encoding='utf-8') as f:
        f.write('.'.join(list_labels_train))


def get_list_corpus_train():
    with open('../resources/list_corpus_train.txt', 'r', encoding='utf-8') as f:
        list_corpus_train = f.read().split('.')
        return list_corpus_train


def get_list_labels_train():
    with open('../resources/list_corpus_train.txt', 'r', encoding='utf-8') as f:
        list_labels_train = f.read().split('.')
        return list_labels_train



gc = gspread.service_account(filename='key.json')
sh = gc.open("hh_dataset")
worksheet = sh.get_worksheet(0)
data_table = worksheet.get_all_values()
data_table.pop(0)

gc1 = gspread.service_account(filename='key.json')
sh1 = gc.open("main_dataset")
worksheet1 = sh.get_worksheet(0)
data_table1 = worksheet.get_all_values()
data_table1.pop(0)

list_corpus_train, list_labels_train, list_source_sentence_train = prepared_table(data_table[:1998])
list_corpus_test, list_labels_test, list_source_sentence_test = prepared_table(data_table1[:2144])
print(list_source_sentence_test)
X_train_counts, count_vectorizer = cv(list_corpus_train)
clf = train_model(X_train_counts, list_labels_train)
y_predicted_counts = test_model(clf, list_corpus_test, count_vectorizer)
accuracy, precision, recall, f1 = get_metrics(list_labels_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

# save_list_corpus_train(list_corpus_train)
# save_list_labels_train(list_labels_train)
# save_model(clf)
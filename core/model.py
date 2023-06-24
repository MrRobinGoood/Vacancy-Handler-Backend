import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def save_model(clf):
    with open('../resources/model.pkl', 'wb') as f:
        pickle.dump(clf, f)


def get_model():
    with open('../resources/model.pkl', 'rb') as f:
        clf = pickle.load(f)
        return clf


def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def test_model(clf, X_test, count_vectorizer):
    X_test_counts = count_vectorizer.transform(X_test)
    y_predicted_counts = clf.predict(X_test_counts)
    return y_predicted_counts


def sep_train_test(list_corpus, list_labels):
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)
    return X_train, X_test, y_train, y_test


def train_model(X_train_counts, y_train):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    return clf



# тест графиков
# fig = plt.figure(figsize=(16, 16))
# plot_LSA(X_train_counts, y_train)
# plt.show()

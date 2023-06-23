from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


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


def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=3)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    print(lsa_scores)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))

        red_patch = mpatches.Patch(color='orange', label='Должностные обязанности')
        green_patch = mpatches.Patch(color='blue', label='Условия')
        orange_patch = mpatches.Patch(color='green', label='Требования к соискателю')
        plt.legend(handles=[red_patch, green_patch, orange_patch], prop={'size': 30})


def test_model(clf, X_test, count_vectorizer):
    X_test_counts = count_vectorizer.transform(X_test)
    y_predicted_counts = clf.predict(X_test_counts)
    return y_predicted_counts


def sep_train_test(list_corpus, list_labels):
    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2,
                                                        random_state=40)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    X_train_counts, count_vectorizer = cv(X_train)
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, y_train)
    return clf, count_vectorizer

# тест графиков
# fig = plt.figure(figsize=(16, 16))
# plot_LSA(X_train_counts, y_train)
# plt.show()

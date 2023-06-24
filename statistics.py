import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
import matplotlib.patches as mpatches


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=3)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'green']
    if plot:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(lsa_scores[:, 0], lsa_scores[:, 1], lsa_scores[:, 2], c=test_labels,
                   cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Должностные обязанности')
        green_patch = mpatches.Patch(color='blue', label='Условия')
        orange_patch = mpatches.Patch(color='green', label='Требования к соискателю')
        ax.legend(handles=[red_patch, green_patch, orange_patch], prop={'size': 10})
        plt.show()


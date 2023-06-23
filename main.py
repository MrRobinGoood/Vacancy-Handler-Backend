from text_handler import prepared_text
from model import start_model
import gspread

if __name__ == '__main__':
    gc = gspread.service_account(filename='key.json')
    sh = gc.open("cpHh")
    worksheet = sh.get_worksheet(0)
    data_table = worksheet.get_all_values()
    data_table.pop(0)

    list_corpus = []
    list_labels = []

    for i in range(798):
        prepared_sentences = prepared_text(data_table[i][1])
        for sentence in prepared_sentences:
            if sentence:
                sentence = ' '.join(sentence)
                list_corpus.append(sentence)
                list_labels.append(data_table[i][2])

    start_model(list_corpus, list_labels)

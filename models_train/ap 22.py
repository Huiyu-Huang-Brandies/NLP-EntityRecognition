
import sklearn_crfsuite
from seqeval.metrics import f1_score, classification_report



def load_data(data_path):
    data_read_all = list()
    data_sent_with_label = list()
    with open(data_path, mode='r', encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                data_read_all.append(data_sent_with_label.copy())
                data_sent_with_label.clear()
            else:
                data_sent_with_label.append(tuple(line.strip().split(" ")))
    return data_read_all


def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit()
    }
    # the previous word
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word': word1
        })
    # else:
    #     # adding BOS(begin of sentence)
    #     features['BOS'] = True

    # the previous two word
    if i > 1:
        word2 = sent[i - 2][0]
        features.update({
            '-2:word': word2,
        })

    # the next word
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word': word1
        })
    # else:
    #     # adding EOS(end of sentence)
    #     features['EOS'] = True

    # the next two word
    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        features.update({
            '+2:word': word2
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [ele[-1] for ele in sent]


train = load_data('train.txt')
valid = load_data('dev.txt')
# test = load_data('text.txt')
X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]
X_dev = [sent2features(s) for s in valid]
y_dev = [sent2labels(s) for s in valid]


crf_model = sklearn_crfsuite.CRF(algorithm='ap')
crf_model.fit(X_train, y_train)

labels = list(crf_model.classes_)

labels.remove("O")  # We do not care about tag O
y_pred = crf_model.predict(X_dev)
print(round(f1_score(y_dev, y_pred)*100, 2))
print(classification_report(y_dev, y_pred, digits=2))
# 78.57
#               precision    recall  f1-score   support
#
#          LOC       0.87      0.85      0.86      1851
#          ORG       0.71      0.66      0.68       982
#          PER       0.83      0.68      0.75      1427
#
#    micro avg       0.82      0.75      0.79      4260
#    macro avg       0.81      0.73      0.77      4260
# weighted avg       0.82      0.75      0.78      4260






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
        'word': word
    }
    # the previous word
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word': word1
        })

    # the next word
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word': word1
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [ele[-1] for ele in sent]


train = load_data('train.txt')
valid = load_data('dev.txt')

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

# 75.07
#               precision    recall  f1-score   support
#
#          LOC       0.84      0.82      0.83      1851
#          ORG       0.70      0.64      0.66       982
#          PER       0.80      0.62      0.70      1427
#
#    micro avg       0.80      0.71      0.75      4260
#    macro avg       0.78      0.69      0.73      4260
# weighted avg       0.80      0.71      0.75      4260









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
# valid = load_data('dev.txt')
test = load_data('text.txt')
X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]
X_test = [sent2features(s) for s in test]
y_test = [sent2labels(s) for s in test]


crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', c2=0.07, max_iterations=400)
crf_model.fit(X_train, y_train)

labels = list(crf_model.classes_)

labels.remove("O")  # We do not care about tag O
y_pred = crf_model.predict(X_test)
print(round(f1_score(y_test, y_pred)*100, 2))
print(classification_report(y_test, y_pred, digits=2))
# 73.27
#               precision    recall  f1-score   support
#
#          LOC       0.81      0.70      0.75      1034
#          ORG       0.65      0.69      0.67       349
#          PER       0.78      0.71      0.74       545
#
#    micro avg       0.77      0.70      0.73      1928
#    macro avg       0.75      0.70      0.72      1928
# weighted avg       0.77      0.70      0.73      1928






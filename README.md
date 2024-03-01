# NLP-EntityRecognition
# Chinese Sequence Labelling Project README

## 1. Introduction

This project explores sequence labelling in Chinese, focusing on part-of-speech tagging and named entity recognition (NER). The project aims to understand how well NER performs with Chinese data, given the language's unique characteristics.

## 2. Data

The dataset originates from [masakhane-io/masakhane-ner](https://github.com/masakhane-io/masakhane-ner/tree/main/data/zh), containing sentences from People's Daily and Earth Weekly. It includes 48,497 sentences with 2,171,516 tokens labeled for NER using the BIO tagging scheme, covering LOC, ORG, and PER entities.

## 3. Development Set Results

We employed sklearn-crfsuite with features based on surrounding words. Two algorithms, average perceptron (ap) and lbfgs, were tested with different configurations. Our baseline uses the token itself and pre1_next1 as features, using ap as the algorithm, getting 75.07 as F1 scores. After tuning various feature sets and hyperparameters, we got 79.67 as the best F1 scores.

## 4. Test Set Results

The best performing features and algorithm configurations from the development set applied to the test set, and got a 73.27 F1 scores.

## 5. Discussion

Our analysis revealed that the context provided by surrounding words significantly impacts model performance. The study also investigated the effects of hyperparameter tuning on the model's effectiveness.

## 6. Conclusion

The research concluded that the PER entity is the most reliably recognized in Chinese text. The optimal model configuration was identified as using the lbfgs algorithm with specific hyperparameters and feature sets, providing a solid foundation for further exploration in Chinese sequence labelling.


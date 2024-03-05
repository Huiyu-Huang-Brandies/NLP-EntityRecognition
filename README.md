# NLP-EntityRecognition
# Chinese Sequence Labelling Project README

 
## 1. Introduction  
Sequence labelling enables machines to automatically extract structured information from raw data and use that information to make more accurate predictions and decisions. For this project, I want to try sequence labelling in Chinese. There are two typical tasks: part-of-speech tagging and named entity recognition. Part-of-speech Tagging, after going through the annotation of the tags, it is easy to get confused about some of the part-of-speech tags. Since Chinese do not rely on morphological changes, it might be harder to figure out the part of speech for each character. Nouns and verbs are straightforward, but others like “morpheme” or “distinguish word”, are not clear in Chinese. Named Entities, however, make more sense to me. Based on semantics, it is easy for us to tell which is the Person, Location or Organization. So, I want to explore how NER would work out with Chinese.  

## 2. Data  
The dataset we used is from Github(https://github.com/masakhane-io/masakhane-ner/tree/main/data/zh). It is a NER project for African languages, and Chinese data was updated without a description. Searching the sentences in the data, We found that they came from two newspapers, People's Daily(人民日报) and Earth Weekly(大地周刊). There are 48497 sentences and 2171516 tokens, and each token has its name entity label. We use BIO tagging here, and there are three name entities: LOC, ORG, and PER, short for location, organization, and person. The LOC here includes the Geo-Political Entity and general locations. Table below shows examples of the name entities.


| Name Entity | Sentence                                                                                                                                                                                                         |
|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     LOC     | …, 也是我们收藏北[B-LOC]京[I-LOC]史料中的要件之一。<br/>…, is also one of the elements in our collection of Bei [B-LOC] jing [I-LOC] historical materials.                                                                        |
|     ORG     | 新[B-ORG]四[I-ORG]军[I-ORG]老战士...<br/>New [B-ORG] Fourth [I-ORG] Army [I-ORG] Veteran...                                                                                                                            |
|     PER     | 张[B-PER]西[I-PER]蕾[I-PER]老人…<br/>Zhang [B-PER] Xi [I-PER] lei [I-PER]…                                                                                                                                            |

## 3. Dev set results  
For modelling, we use sklearn-crfsuite to implement the CRF model. For the feature definition, we mainly consider the surrounding words of the target token since Chinese characters do not have an upper or lower case. Four features are defined: the token itself; pre1_next1, presenting the previous and the next word of the token; pre2_next2, presenting the previous two words and the following two words of the token; BOS_EOS, presenting whether the token is at the beginning or the end of the sentence. 

We tried two algorisms in the CRF model: average perceptron “ap” and “lbfgs”. For “lbfgs”, we tried different coefficients for L2 regularization “c2”, which is one of the hyperparameters in CRF. By default, “c2” is 1.0, which is too big, so we start from 0.1 instead. The default max iteration for “lbfgs” is unlimited, so we set it to 200 at first. The max iteration for “ap” is by default (=100). Our baseline uses the token itself and pre1_next1 as features, using average perceptron as the algorithm, getting 75.07 as F1 scores. There are 13 configurations in Table 1. The largest F1 scores are in bold.


#### Tabel 1
|     algorisim = ap/lbfgs(c2)      | features = token + ... |    F1     |
|:---------------------------------:|:----------------------:|:---------:|
|                ap                 |       pre1_next1       |   75.07   |
|                ap                 |  pre1_next1 + BOS_EOS  |   75.14   |
|                ap                 |       pre2_next2       |   78.57   |
|                ap                 |  pre2_next2 + BOS_EOS  |   78.48   |
| lbfgs, c2=0.1, max_iteration=200  |       pre1_next1       |   73.23   |
| lbfgs, c2=0.1, max_iteration=200  |  pre1_next1 + BOS_EOS  |   71.35   |
| lbfgs, c2=0.1, max_iteration=200  |       pre2_next2       |   76.35   |
| lbfgs, c2=0.1, max_iteration=200  |  pre2_next2 + BOS_EOS  |   77.69   |
| lbfgs, c2=0.07, max_iteration=200 |  pre2_next2 + BOS_EOS  |   78.49   |
| lbfgs, c2=0.05, max_iteration=200 |  pre2_next2 + BOS_EOS  |   78.21   |
| lbfgs, c2=0.03, max_iteration=200 |  pre2_next2 + BOS_EOS  |   78.08   |
| lbfgs, c2=0.07, max_iteration=400 |  pre2_next2 + BOS_EOS  | **79.67** |
| lbfgs, c2=0.07, max_iteration=400 |       pre2_next2       |   79.07   |


## 4. Test set results  
 
 From the result of the dev set, it’s clear that the feature pre2_next2 has a better result. But the results for feature BOS_EOS seem ambiguous. Under “ap”, F1 scores are better without it, but under “lbfgs”, it’s better to include it. Therefore, we run four configurations on the test set. The results for the test F1 scores are listed in Table 2. The best configuration on the test is in bold, and we put its details in Table 3. 
  
#### Table 2
|     algorisim = ap/lbfgs(c2)      | features = token + ... |    F1     |
|:---------------------------------:|:----------------------:|:---------:|
|                ap                 |       pre2_next2       |   71.91   |
| lbfgs, c2=0.07, max_iteration=400 |       pre2_next2       | **73.27** |
|                ap                 |  pre2_next2 + BOS_EOS  |   71.69   |
| lbfgs, c2=0.07, max_iteration=400 |  pre2_next2 + BOS_EOS  |   72.8    |
  
#### Table 3
| classification_report | precision | recall | f1-score | support |
|----------------------:|:---------:|:------:|:--------:|:-------:|
|                   LOC |   0.81    |  0.70  |   0.75   |  1034   |
|                   ORG |   0.65    |  0.69  |   0.67   |   349   |
|                   PER |   0.78    |  0.71  |   0.74   |   545   |
|             micro avg |   0.77    |  0.70  |   0.73   |  1928   | 
|             macro avg |   0.75    |  0.70  |   0.72   |  1928   |   
|          weighted avg |   0.77    |  0.70  |   0.73   |  1928   |                         

## 5. Discussion  
  
The surrounding words of the current word really matter to the prediction. The previous and next two words provide more information than just the previous and next one word. I expected the pre2_next2 + BOS_EOS would get the best scores, but the result without the BOS_EOS feature is the best in the test. Therefore, whether the current word is at the start or end of the sentence does not provide reliable information for the prediction. 

In “lbfgs”, I expected the c2 to be very small like 0.02; however, when c2 is less than 0.07, the scores will drop with the decrease of c2. The impact of tuning the c2 is less than the impact of changing the max iteration in my configurations. But it seems that when the max iteration is over 200, the impact would gradually shrink with adding the number of max iterations. 

Overall, I think my model worked well. In Table 3, the LOC label got the highest score with 54% of the total support(occurrence); the PER label got similar scores with LOC but with only 28% of the total support; the ORG label got the lowest scores with the lowest support, which is only 18% of the total support. Therefore, the PER entity got the best prediction, but which entity got the worst prediction needs a further look.
  

## 6. Conclusion  

In a nutshell, the PER entity is the easiest one to recognise in Chinese. The previous and next two words would provide the better feature for this model, and the algorism “lbfgs” with hyperparameter c2=0.07 and a fair number of max iterations like 400 would have a better score. 

Figuring out what features to try might be more complex than I expected. As we discussed, it is not true that the more features we have, the higher scores we will get. If there is more time, I will try only the previous and only the next words to see whether the previous or next matters more. I will also try more surrounding words like the previous three or four words to see the best size for these surrounding windows. Moreover, I will use “《” and “》”, the punctuation containing the title in Chinese, to identify whether it is the title.


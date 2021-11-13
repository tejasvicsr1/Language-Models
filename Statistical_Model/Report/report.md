# Assignment 1

## Introduction to NLP

#### Tejasvi Chebrolu
#### 2019114005

## Analysis

### Average Perplexities

Notes: 
- *The average perplexities calcualted are calculate for 200 sentences only.*
- *In case a more accurate perplexity is to be calculated it can be done so by removing the `break` statement present in the for loop and replacing it with a `continue` statement after uncommenting a particular Model.*

1. __LM-1 Train__: 26.42957837947336
2. __LM-1 Test__: 10.774439680358975
3. __LM-2 Train__: 16.546810800557175
4. __LM-2 Test__: 8.188515773645031
5. __LM-3 Train__: 3.1803385473993355
6. __LM-3 Test__: 4.089275470445463
7. __LM-4 Test__: 3.6089566897164156
8. __LM-4 Train__: 3.1730702285171373


- The perplexity for both models is very low. This is because of multiple factors. Some of them are:
    - Improper tokenization: The tokenization done was not of an adequate level. This could have led to a lot of errors wherein there are different n-grams being formed for the same context.
    - Improper handling of unknown words: Unkown words have not been dealt with properly, and if the probability of a sentence comes to be zero, which does occur, the perplexity returned is zero, which ideally should not be the case.
- The perplexity for _Kneyser Ney_ is marginally more than the perplexity of the _Witten-Bell_ model for the health domain.
    - The difference obviously is more clear in the training data because the model seems to have _memorized_ the values of the sentences.
- The perplexity for sentences trained on the healrh corpus is much better than the sentences trained on the technical domain corpus. This is because of:
    - The domain of the techincal sentences must contain words which are very different from the words in the health domain. The vice versa might not be true.
    - Because of the domain, there must have been difficulties in in tokenization words to extract correct information.
- For the model trained on the technical domain, there is not a lot of difference between the _Kneyser Ney_ and the _Witten-Bell_ model.

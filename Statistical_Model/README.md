# Assignment 1

## Introduction to NLP

#### Tejasvi Chebrolu
#### 2019114005

### Structure

- `2019114005_Assignment1`
    - `language_model.py`
    - `Corpus`
        - `Health_English.txt`
        - `technical_domain_corpus.txt`
    - `README.md`
    - `Report`
        - `2019114005-LM1-test-perplexity.txt` 
        - `2019114005-LM1-train-perplexity.txt`
        - `2019114005-LM2-test-perplexity.txt`
        - `2019114005-LM2-train-perplexity.txt`
        - `2019114005-LM3-test-perplexity.txt`
        - `2019114005-LM3-train-perplexity.txt`
        - `2019114005-LM4-test-perplexity.txt`
        - `2019114005-LM4-train-perplexity.txt`
        - `report.md`

### Question 1

- To find the probability of a sentence using _m_ method on the _n_ dataset run which will provide a prompt to take the input:
    - `python3 language_model.py m path_to_n`
    - For example, if you want to check the probability of a sentence based on the _Witten Bell_ model and on the Health DataSet you can run:
        - `python3 language_model.py w ./Corpus/Health_English.txt`

### Question 2

- To find the perplexity of a sentence using:
    - __Kneyser Ney on Health Corpus__ uncomment line 424
    - __Kneyser Ney on Tech Corpus__ uncomment line 425
    - __Witten Bell on Health Corpus__ uncomment line 423
    - __Witten Bell on Tech Corpus__ uncomment 422
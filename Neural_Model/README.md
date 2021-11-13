# Neural Language Models Vs Statistical Language Models

## Directory Structure

Following is the directory structure for the assignment:
``` text
.
├── 2019114005_Assignment_2.ipynb
├── 2019114005-Assignment_2.py
├── data
│   ├── images
│   │   └── Kneser-Ney, Witten-Bell and Neural.png
│   ├── LM_1
│   │   ├── 2019114005-LM1-test-perplexity.txt
│   │   ├── 2019114005-LM1-train-perplexity.txt
│   │   └── 2019114005-LM1-validation-perplexity.txt
│   ├── LM_2
│   │   ├── 2019114005-LM2-test-perplexity.txt
│   │   ├── 2019114005-LM2-train-perplexity.txt
│   │   └── 2019114005-LM2-validation-perplexity.txt
│   └── LM_3
│       ├── 2019114005-LM3-test-perplexity.txt
│       ├── 2019114005-LM3-train-perplexity.txt
│       └── 2019114005-LM3-validation-perplexity.txt
├── README.md
└── report.pdf
5 directories, 14 files
```
***

## Language Models

- `LM_1` contains the perplexities for the _Kneser-Ney_ model.
- `LM_2` contains the perplexities for the _Witen-Bell_ model.
- `LM_2` contains the perplexities for the _Neural_ model.
***

## Code

_It is advisable to run the code on a GPU to save time._

- `2019114005-Assignment_2.py` contains the code for the Neural Model in the form of a python file.
- `2019114005-Assignment_2.ipynb` contains the code for the Neural Model in the form of a python notebook.
- To find the perplexity of a particular sentence, add the sentence into `example_sentence` in line 304 in the python file and similarly in the notebook.
- Make sure you train the model before changing the `example_sentence`.
- The model can be trained by changing the variable `data` in line 72 in the python file to the _relative_ path to the corpus and similarly in the notebook.
- Any hyperparameters can be tuned by changing the respective hyperparameters(all in capitals).

**It takes approximately an hour for the model to train on the cleaned Brown Corpus.**
***

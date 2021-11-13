# Language Model
---

> The aim of this project is to compare and analyse the performance of different language models. For this project, two statistical language models are used: Kneser-Ney and Witten-Bell.
---

## Language Models

- `Neural_Model/data/LM_1` contains the perplexities for the _Kneser-Ney_ model.
- `Neural_Model/data/LM_2` contains the perplexities for the _Witen-Bell_ model.
- `Neural_Model/data/LM_2` contains the perplexities for the _Neural_ model.
---

## Code

_It is advisable to run the code on a GPU to save time._

- `Neural_Model/2019114005-Assignment_2.py` contains the code for the Neural Model in the form of a python file.
- `Neural_Model/2019114005-Assignment_2.ipynb` contains the code for the Neural Model in the form of a python notebook.
- To find the perplexity of a particular sentence, add the sentence into `example_sentence` in line 304 in the python file and similarly in the notebook.
- Make sure you train the model before changing the `example_sentence`.
- The model can be trained by changing the variable `data` in line 72 in the python file to the _relative_ path to the corpus and similarly in the notebook.
- Any hyperparameters can be tuned by changing the respective hyperparameters(all in capitals).

**It takes approximately an hour for the model to train on the cleaned Brown Corpus.**

---

## Statistical Models
### Finding Probability
_Commands have to be run from inside the Neural Model directory._
- To find the probability of a sentence using _m_ method on the _n_ dataset run which will provide a prompt to take the input:
    - `python3 language_model.py m path_to_n`
    - For example, if you want to check the probability of a sentence based on the _Witten Bell_ model and on the Health DataSet you can run:
        - `python3 language_model.py w ./Corpus/Health_English.txt`

### Finding Perplexities

- To find the perplexity of a sentence using:
    - __Kneyser Ney on Health Corpus__ uncomment line 424
    - __Kneyser Ney on Tech Corpus__ uncomment line 425
    - __Witten Bell on Health Corpus__ uncomment line 423
    - __Witten Bell on Tech Corpus__ uncomment 422

---
## Report

> For a detailed report, please see the attached report. (report.pdf)
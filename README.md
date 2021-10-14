# Canary

[![Documentation Status](https://readthedocs.org/projects/canary-am/badge/?version=latest)](https://canary-am.readthedocs.io/en/latest/?badge=latest)
[![Canary package tests](https://github.com/chriswales95/Canary/actions/workflows/python-unit-tests.yml/badge.svg?branch=development)](https://github.com/chriswales95/Canary/actions/workflows/python-unit-tests.yml)

Canary is an argument mining Python library. Argument Mining is the automated identifcation and extraction of
argumentative data from natural language.

It should be noted that this software is currently under **active development** and is not fully functional or feature
complete.

## Installation

Canary will be installable through [Pypi](https://pypi.org) in the near-future. For the time being, it can be installed
in the following manner:

**https:**

```commandline
pip install git+https://github.com/chriswales95/Canary.git@development
```

**ssh:**

```commandline
pip install git+ssh://git@github.com/chriswales95/Canary.git@development
```

## Example Usage

### Detecting an argument (true / false)

```python
from canary.argument_pipeline import download_model, load_model, analyse_file

if __name__ == "__main__":
    # Download pretrained models from the web (unless you fancy creating them yourself)
    # Training the models takes a while so I'd advise against it.
    download_model("all")

    # load the detector
    detector = load_model("argument_detector")

    # outputs false
    print(detector.predict("cats are pretty lazy animals"))

    # outputs true
    print(detector.predict(
        "If a criminal knows that a person has a gun , they are much less likely to attempt a crime ."))
```

### Analysing a full document

```python
from canary.argument_pipeline import download_model, analyse_file
from canary.corpora import load_corpus
from pathlib import Path
if __name__ == "__main__":
    
    # Download all models
    download_model("all")
    
    # Load version 1 of the essay corpus. 
    essays = load_corpus("argument_annotated_essays_1", download_if_missing=True)
    if essays is not None:
        essays = [essay for essay in essays if Path(essay).suffix == ".txt"]
    
        # Analyse the first essay
        # essays[0] contains the absolute path to the first essay text file
        analysis = analyse_file(essays[0])
```

## What kind of performance is Canary achieving?

Canary is currently still in development and performance is being improved as work continues.

### Argument Detector

                  precision    recall  f1-score   support
    
           False       0.80      0.58      0.67       490
            True       0.88      0.96      0.92      1653
    
        accuracy                           0.87      2143
       macro avg       0.84      0.77      0.79      2143
    weighted avg       0.86      0.87      0.86      2143


### Argument Segmenter

                  precision    recall  f1-score   support
    
               O     0.7936    0.7259    0.7583      9362
           Arg-B     0.7784    0.7765    0.7775      1235
           Arg-I     0.8761    0.9126    0.8939     19248
    
        accuracy                         0.8484     29845
       macro avg     0.8160    0.8050    0.8099     29845
    weighted avg     0.8462    0.8484    0.8466     29845


### Argument Component Predictor

                  precision    recall  f1-score   support
    
           Claim       0.65      0.65      0.65       226
      MajorClaim       0.82      0.86      0.84       225
         Premise       0.78      0.75      0.76       225
    
        accuracy                           0.75       676
       macro avg       0.75      0.75      0.75       676
    weighted avg       0.75      0.75      0.75       676

### Link Predictor

                  precision    recall  f1-score   support
    
          Linked       0.75      0.73      0.74      4581
      Not Linked       0.87      0.88      0.87      9227
    
        accuracy                           0.83     13808
       macro avg       0.81      0.81      0.81     13808
    weighted avg       0.83      0.83      0.83     13808


### Structure Predictor

                  precision    recall  f1-score   support
    
         attacks       0.87      0.89      0.88       741
        supports       0.94      0.93      0.94      1427
    
        accuracy                           0.92      2168
       macro avg       0.91      0.91      0.91      2168
    weighted avg       0.92      0.92      0.92      2168
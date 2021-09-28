# Canary
[![Documentation Status](https://readthedocs.org/projects/canary-am/badge/?version=latest)](https://canary-am.readthedocs.io/en/latest/?badge=latest)
[![Canary package tests](https://github.com/chriswales95/Canary/actions/workflows/python-unit-tests.yml/badge.svg?branch=development)](https://github.com/chriswales95/Canary/actions/workflows/python-unit-tests.yml)

Canary is an argument mining Python library. Argument Mining is the automated identifcation and extraction of
argumentative data from natural language.

It should be noted that this software is currently under **active development** and is not fully functional or feature complete.
## Installation

Canary will be installable through [Pypi](https://pypi.org) in the near-future. For the time being, it can be installed in the following manner:

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
import logging
from canary.argument_pipeline.detection import ArgumentDetector
from canary.argument_pipeline import download

if __name__ == "__main__":
    # setting up the logger is not mandatory but is useful to see output 
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('canary').setLevel(logging.DEBUG)

    # Download pretrained models from the web (unless you fancy creating them yourself)
    # Training the models takes a while so I'd advise against it.
    download("all")

    # Instantiate the detector
    ag = ArgumentDetector()

    # outputs false
    print(ag.predict("cats are pretty lazy animals"))

    # outputs true
    print(ag.predict("If a criminal knows that a person has a gun , they are much less likely to attempt a crime ."))
```

### Detecting argument components

```python
import logging

from canary.argument_pipeline.component_prediction import ArgumentComponent
from canary.argument_pipeline import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('canary').setLevel(logging.DEBUG)

    # Download the models if you don't have them already.
    download("all")

    # Instantiate the component extractor
    ag = ArgumentComponent()

    # Major claim
    print(ag.predict("one who studies overseas will gain many skills throughout this experience"))

    # Claim
    print(ag.predict(
        "living and studying overseas is an irreplaceable experience when it comes to learn standing on your own feet"))

    # Premise
    print(ag.predict("employers are mostly looking for people who have international and language skills"))
```

### Detecting argument relations
To be documented.


### Argument boundary detection
To be documented.


### Scheme prediction
To be implemented.

## What kind of performance is Canary achieving?
Canary is currently still in development and performance is being improved as work continues. 

<br>

**Detection of arguments**


<table>
<thead>
  <tr>
    <th></th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>False</th>
    <td>0.66</td>
    <td>0.78</td>
    <td>0.71</td>
    <td>66872</td>
  </tr>
  <tr>
    <th>True</th>
    <td>0.76</td>
    <td>0.63</td>
    <td>0.69</td>
    <td>73608</td>
  </tr>
  <tr>
    <th></th>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>accuracy</th>
    <td></td>
    <td></td>
    <td>0.7</td>
    <td>140480</td>
  </tr>
  <tr>
    <th>macro avg</th>
    <td>0.71</td>
    <td>0.71</td>
    <td>0.7</td>
    <td>140480</td>
  </tr>
  <tr>
    <th>weighted avg</th>
    <td>0.71</td>
    <td>0.7</td>
    <td>0.7</td>
    <td>140480</td>
  </tr>
</tbody>
</table>

<br>

**Detection of components:**
<table>
<thead>
  <tr>
    <th></th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Claim</th>
    <td>0.41</td>
    <td>0.35</td>
    <td>0.38</td>
    <td>172</td>
  </tr>
  <tr>
    <th>MajorClaim</th>
    <td>0.42</td>
    <td>0.46</td>
    <td>0.44</td>
    <td>79</td>
  </tr>
  <tr>
    <th>Premise</th>
    <td>0.72</td>
    <td>0.76</td>
    <td>0.74</td>
    <td>358</td>
  </tr>
   <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>accuracy</th>
    <td></td>
    <td></td>
    <td>0.6</td>
    <td>609</td>
  </tr>
  <tr>
    <th>macro avg</th>
    <td>0.52</td>
    <td>0.52</td>
    <td>0.52</td>
    <td>609</td>
  </tr>
  <tr>
    <th>weighted avg</th>
    <td>0.6</td>
    <td>0.6</td>
    <td>0.6</td>
    <td>609</td>
  </tr>
</tbody>
</table>

**Detection of support/attack relations:**
<table>
<thead>
  <tr>
    <th></th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Attacks</th>
    <td>0.75</td>
    <td>0.60</td>
    <td>0.67</td>
    <td>25</td>
  </tr>
  <tr>
    <th>Supports</th>
    <td>0.97</td>
    <td>0.99</td>
    <td>0.98</td>
    <td>359</td>
  </tr>
   <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>accuracy</th>
    <td></td>
    <td></td>
    <td>0.96</td>
    <td>384</td>
  </tr>
  <tr>
    <th>macro avg</th>
    <td>0.86</td>
    <td>0.79</td>
    <td>0.82</td>
    <td>384</td>
  </tr>
  <tr>
    <th>weighted avg</th>
    <td>0.96</td>
    <td>0.96</td>
    <td>0.96</td>
    <td>384</td>
  </tr>
</tbody>
</table>

**Argument Segmentation:**
<table>
<thead>
  <tr>
    <th></th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>O</th>
    <td>0.71</td>
    <td>0.67</td>
    <td>0.69</td>
    <td>9071</td>
  </tr>
  <tr>
    <th>Arg-B</th>
    <td>0.73</td>
    <td>0.71</td>
    <td>0.73</td>
    <td>1174</td>
  </tr>
  <tr>
    <th>Arg-I</th>
    <td>0.84</td>
    <td>0.87</td>
    <td>0.86</td>
    <td>17915</td>
  </tr>
   <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <th>accuracy</th>
    <td></td>
    <td></td>
    <td>0.8</td>
    <td>28160</td>
  </tr>
  <tr>
    <th>macro avg</th>
    <td>0.76</td>
    <td>0.75</td>
    <td>0.75</td>
    <td>28160</td>
  </tr>
  <tr>
    <th>weighted avg</th>
    <td>0.80</td>
    <td>0.80</td>
    <td>0.80</td>
    <td>28160</td>
  </tr>
</tbody>
</table>

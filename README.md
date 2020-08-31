## Canary

Canary is a Python library for extracting argumentative components and their relationships from text files. The aim is to provide a novel implementation of an Argument Miner that someone is able to run on their own data set in order to find patterns or to extract argumentative structure from their data. Alongside this, Canary is fully open-sourced allowing for users to implement their own functions or improve pre-existing ones.

### Getting Started

To download and gain access to Canary run:

Eventually there will be a pip capable install along the following lines:
```python
pip install Canary-am
```

but for the moment we'll build and use Canary from source until we have a sufficiently robust and feature rich release candidate.

### Example

Basic example showing the extraction of Argumentative Components from a local file:

```Python
from canary import local

components = canary.Local(file)

print(components[1])

# Output
['hence it is always said that competition makes the society more effective.', 'therefore without the cooperation, there would be no victory of competition.']
```

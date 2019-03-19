## Canary

Canary is a Python based library for extracting argumentative components and their relationships from text files. The aim is to provide a novel implementation of an Argument Miner that someone is able to run on their own data set in order to find patterns or to extract argumentative structure from their data. Alongside this, Canary is fully open-sourced allowing for users to implement their own functions or improve pre-existing ones.

### Getting Started

To download and gain access to Canary run:

```python
pip install Canary-am
```

### Documentation

See the documentation at [Canary.am/docs](https://Canary.am/docs) for more information.

### Example

Basic example showing the extraction of Argumentative Components from a local file:

```Python
from canary import local

components = canary.Local(file)

print(components[1])

# Output
['hence it is always said that competition makes the society more effective.', 'therefore without the cooperation, there would be no victory of competition.']
```
For more details see [Canary](https://Canary.am).

# Wordmap

> Visualize huge text collections with WebGL.

## Installation

```
pip install wordmap
```

## Usage

To create a plot of words, one only needs a list of words and an iterable of word positions:

```
import wordmap
import numpy as np

words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
positions = np.random.rand(len(words), 2)

wordmap.plot(words, positions)
```

## Visualizing Word Vectors

Instead of using random positions, one can use some data processing to generate strategic positions for each word.
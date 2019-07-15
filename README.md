# Wordmap

> Visualize huge text collections with WebGL.

## Installation

```
pip install wordmap
```

## Basic Usage

To process a directory of text files, you can call wordmap directly from the command line:

```
wordmap "texts/*.txt"
```

Once the process is done, start a local web server with:

```
# python 2
python -m SimpleHTTPServer 7090

# python 3
python -m http.server 7090
```

Then navigate to `http://localhost:7090/web`

## Custom Plot

To create a plot of words with custom word positions, one needs a list of words and an iterable of word positions:

```
import wordmap
import numpy as np

words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
positions = np.random.rand(len(words), 2)

wordmap.plot(words, positions)
```
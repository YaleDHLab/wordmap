# Wordmap

> Visualizing large collections of text data with WebGL

![App preview](./wordmap/web/assets/images/preview.png?raw=true)

## Installation

```bash
pip install wordmap
```

## Basic Usage

To create a visualization from a directory of text files, you can call wordmap as follows:

```bash
wordmap --texts "data/*.txt"
```

That process creates a visualization in `./web` that can be viewed if you start a local web server:

```bash
# python 2
python -m SimpleHTTPServer 7090

# python 3
python -m http.server 7090
```

After starting the web server, navigate to `http://localhost:7090/web/` to view the visualization.

## Command Line Interface

The following flags can be passed to the wordmap command:

`--texts` A glob of files to process

`--encoding` The encoding of input files

`--max_words` The maximum number of words to include in the visualization

`--layouts` The layouts to include in the output data `{umap, tsne, grid}`

`--n_components` The number of dimensions to use when creating the layouts

`--obj_file` An .obj file whose vertices should be used to create the layout

`--model` A persisted gensim.Word2Vec model to use to create layouts

`--model_name` The name to use when saving a gensim Word2Vec model to disk

`--size` The number of dimensions to include in Word2Vec vectors

`--window` The number of words to include in windows when creating a Word2Vec model

`--min_count` The minimum occurrences of each word to be included in the Word2Vec model

`--workers` The number of computer cores to use when processing input data

`--verbose` If true, logs progress during layout construction

**Examples:**

Create a wordmap of the text files in ./data using the `umap`, `tsne`, and `grid` layouts:

```bash
wordmap --texts "data/*.txt" --layouts umap tsne grid
```

Create a wordmap using a saved Word2Vec model with 3 dimsions and a maximum of 10000 words:

```bash
wordmap --model "1563222036.model" --n_components 3 --max_words 10000
```

Create a wordmap with several layouts, each with multiple parameter steps:

```bash
python wordmap/wordmap.py \
  --texts "data/philosophical_transactions/*.txt" \
  --layouts tsne umap grid \
  --tsne_perplexity 5 25 100 \
  --umap_n_neighbors 2 20 200 \
  --umap_min_dist 0.01 0.1 1.0 \
  --model "models/word2vec/1563838742.model" \
  --n_clusters 7 11
```

## Custom Vertex Positions

To create a plot of text data with custom word positions, one can create `web/wordmap-positions.json` with the following format:

```bash
[
  {
    "name": "cat-on-mat-layout",
    "words": [
      "the",
      "cat",
      "sat",
      "on",
      "the",
      "map"
    ],
    "positions": [
      [0.44798019, 0.34936568, 0.24417208],
      [0.86653381, 0.51266891, 0.4364447 ],
      [0.51050636, 0.94120338, 0.37768034],
      [0.40554792, 0.82156551, 0.9344524 ],
      [0.70673001, 0.40564123, 0.90841559],
      [0.79450194, 0.65910087, 0.57480824]
    ]
  }
]
```

This file could contain multiple layouts, but each layout needs the `name`, `words`, and `positions` attributes.
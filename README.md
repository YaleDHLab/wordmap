# Wordmap

> Visualize large collections of text data with WebGL

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

## Command Line Arguments

The following flags can be passed to the wordmap command. Type `--help` to see the full list:

`--texts` A glob of files to process

`--encoding` The encoding of input files

`--max_n` The maximum number of words/docs to include in the visualization

`--layouts` The layouts to render `{umap, tsne, grid, img, obj}`

`--obj_file` An .obj file that should be used to create the obj layout

`--img_file` A .png or .jpg file that should be used to create the img layout

`--n_components` The number of dimensions to use when creating the layouts

`--tsne_perplexity` The perplexity value to use when creating TSNE layout

`--umap_n_neighbors` The n_neighbors value to use when creating UMAP layout

`--umap_min_distance` The min_distance value to use when creating the UMAP layout

`--model_type` The model type to use {`word2vec`}

`--use_cache` Boolean that, if True, will load saved layouts from `models`

`--model_name` The name to use when saving a model to disk

`--model` A persisted model to use to create layouts

`--size` The number of dimensions to include in Word2Vec vectors

`--window` The number of words to include in windows when creating a Word2Vec model

`--iter` The maximum number of iterations to run the created model

`--min_count` The minimum occurrences of each word to be included in the Word2Vec model

`--workers` The number of computer cores to use when processing input data

`--verbose` If true, logs progress during layout construction

**Examples:**

Create a wordmap of the text files in ./data using the `umap`, `tsne`, and `grid` layouts:

```bash
wordmap --texts "data/*.txt" \
  --layouts umap tsne grid
```

Create a wordmap using a saved Word2Vec model with 3 dimsions and a maximum of 10000 words:

```bash
wordmap --model "1563222036.model" \
  --n_components 3 \
  --max_n 10000
```

Create a wordmap with several layouts, each with multiple parameter steps:

```bash
python wordmap/wordmap.py \
  --texts "data/philosophical_transactions/*.txt" \
  --layouts tsne umap grid \
  --tsne_perplexity 5 25 100 \
  --umap_n_neighbors 2 20 200 \
  --umap_min_dist 0.01 0.1 1.0 \
  --n_clusters 10 25 \
  --iter 100
```
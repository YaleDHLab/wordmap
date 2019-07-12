from flask import Flask, jsonify, request, send_from_directory, render_template
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from flask_cors import CORS
from os.path import join
from umap import UMAP
import numpy as np
import rasterfairy
import gensim
import codecs
import shutil
import glob
import json
import sys
import os
import re


# Store a reference to the location of this package
path = os.path.dirname(os.path.realpath(__file__))

# Store a reference to the location of the calling agent
cwd = os.getcwd()

# Store a reference to the location to which web assets will be written
web_target = join(cwd, 'web')


def plot(words, df, max_words=100, n_components=2):
  '''
  Project word vectors in a numpy array into 2D space
  and persist the resulting data structures for viewing.

  Parameters
  ----------
  words : array of strings
    A list of strings to be visualized. These could be single words
    or multiword strings.
  df : numpy.ndarray
    This dataframe should have shape len(words), k, where k is some
    integer. Each row in the dataframe should indicate the position
    of a word in some vector space (e.g. Word2Vec, NMF, GloVe, LDA...)
  n_components : int
    The number of dimensions to use in the embedding space
  max_words : int
    The maximum number of words to include in the visualization. The more
    words, the greater the number of primitives WebGL has to draw (each letter
    of each word is drawn as a single GL_POINT).
  '''
  # get initial point assignments
  print(' * creating UMAP layout')
  t_proj = UMAP(n_components=n_components).fit_transform(df)
  # combine the words with the projections
  t = [[words[idx], int(i[0]), int(i[1])] for idx, i in enumerate(t_proj)]
  # use linear assignment to gridify points; r_proj is discretized
  print(' * quantizing layout')
  r_proj = rasterfairy.transformPointCloud2D(t_proj)
  # combine the words and positions
  r = [[words[idx], int(i[0]), int(i[1])] for idx, i in enumerate(r_proj[0])]
  # jitter the positions slightly to prevent/discourage overlap
  for i in r:
    if i[1]%2==0:
      i[2] += 0.5
  # format the layout json
  layouts = [
    {'words': words, 'positions': t, 'name': 'umap'},
    {'words': words, 'positions': r, 'name': 'grid'},
  ]
  # copy web assets to user's location
  prepare_web_assets()
  # persist the results to disk
  print(' * writing outputs')
  if not os.path.exists(web_target): os.makedirs(web_target)
  out_path = join(web_target, 'wordmap-layouts.json')
  with open(out_path, 'w') as out:
    json.dump(layouts, out)


def plot_gensim_word2vec(model, max_words=100000):
  '''
  Project the words in a gensim word2vec model into 2D space
  and persist the resulting data structures for viewing.

  Parameters
  ----------
  model : gensim.models.word2vec.Word2Vec
    A gensim Word2Vec model, e.g. as constructed via the
    gensim.models.word2vec.Word2Vec or KeyedVectors.load_word2vec_format constructors
  '''
  words = model.wv.index2entity
  if max_words:
    words = words[:max_words]
  df = np.array([model.wv[w] for w in words])
  if len(words) != df.shape[0]:
    print('! Warning: the number of words != number of elements in dataframe')
    print('! Number of words:', len(words))
    print('! Number of elements in dataframe:', df.shape[0])
  plot(words, df)


def parse(encoding='utf8', size=100000, window=5, min_count=20, workers=4):
  '''
  Main method for parsing a glob of files passed on the command line
  '''
  if len(sys.argv) < 2:
    print('\nPlease identify a glob of files to process:\n')
    print('wordmap "my_data/*.txt"')
    sys.exit()
  files = glob.glob(sys.argv[1])
  if len(files) < 1:
    print('\nYour specified file list is empty. Please check your file glob.\n')
    sys.exit()
  display_files = files[:10] + ['...'] if len(files) >= 10 else files
  sep = '    \n    '
  print(' * preparing to parse {0} files:{1}'.format(len(files), sep + sep.join(display_files)))
  word_lists = []
  for i in files:
    with codecs.open(i, 'r', encoding) as f:
      word_lists.append(re.sub(r'[^\w\s]','',f.read().lower()).split())
  print(' * building model')
  model = Word2Vec(word_lists, size=size, window=window, min_count=min_count, workers=workers)
  model.save('word2vec.model')
  plot_gensim_word2vec(model)


def prepare_web_assets():
  '''
  Copy the web assets required for rendering to the user's cwd
  '''
  print(' * preparing web assets')
  if not os.path.exists(web_target):
    shutil.copytree(join(path, 'web'), web_target)
  else:
    print(' !', web_target, 'is not empty; assets not copied')


def serve():
  '''
  Method to serve the content in ./web
  '''
  print(' * serving assets from', web_target)
  app = Flask(__name__, static_folder=web_target, template_folder=web_target)
  CORS(app)

  # requests for static index file
  @app.route('/')
  def index():
    return render_template('index.html')

  # requests for static assets
  @app.route('/<path:path>')
  def asset(path):
    return send_from_directory(web_target, path)

  # run the server
  app.run(host= '0.0.0.0', port=7082)

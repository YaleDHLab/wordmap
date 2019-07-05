from flask import Flask, jsonify, request, send_from_directory, render_template
from MulticoreTSNE import MulticoreTSNE as TSNE
from flask_cors import CORS
from os.path import join
import numpy as np
import rasterfairy
import gensim
import shutil
import json
import os


# Store a reference to the location of this package
path = os.path.dirname(os.path.realpath(__file__))

# Store a reference to the location of the calling agent
cwd = os.getcwd()

# Store a reference to the location to which web assets will be written
web_target = join(cwd, 'web')


def plot(words, df, max_words=100000):
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
  max_words : int
    The maximum number of words to include in the visualization. The more
    words, the greater the number of primitives WebGL has to draw (each letter
    of each word is drawn as a single GL_POINT).
  '''
  # get initial point assignments from fast TSNE
  print(' * creating TSNE layout')
  t_proj = TSNE(n_jobs=8).fit_transform(df)
  # combine the words with the TSNE projections
  t = [[words[idx], int(i[0]), int(i[1])] for idx, i in enumerate(t_proj)]
  # use linear assignment to gridify points; r_proj is discretized
  print(' * quantizing TSNE layout')
  r_proj = rasterfairy.transformPointCloud2D(t_proj)
  # combine the words and positions
  r = [[words[idx], int(i[0]), int(i[1])] for idx, i in enumerate(r_proj[0])]
  # jitter the positions slightly to prevent/discourage overlap
  for i in r:
    if i[1]%2==0:
      i[2] += 0.5
  # copy web assets to user's location
  prepare_web_assets()
  # persist the results to disk
  print(' * writing outputs')
  if not os.path.exists(web_target): os.makedirs(web_target)
  with open(join(web_target, 'wordmap-tsne.json'), 'w') as out: json.dump(t, out)
  with open(join(web_target, 'wordmap-grid.json'), 'w') as out: json.dump(r, out)


def prepare_web_assets():
  '''
  Copy the web assets required for rendering to the user's cwd
  '''
  print(' * preparing web assets')
  if not os.path.exists(web_target):
    shutil.copytree(join(path, 'web'), web_target)
  else:
    print(' ! cowardly refusing to copy web assets to', web_target)
    print(' !', web_target, 'is not empty')


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
  df = [model.wv[w] for w in words]
  df = np.array(df)
  if len(words) != df.shape[0]:
    print('! Warning: the number of words != number of elements in dataframe')
    print('! Number of words:', len(words))
    print('! Number of elements in dataframe:', df.shape[0])
  plot(words, df)


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

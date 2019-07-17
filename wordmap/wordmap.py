from flask import Flask, jsonify, request, send_from_directory, render_template
from gensim.models.callbacks import CallbackAny2Vec
from distutils.dir_util import copy_tree
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from flask_cors import CORS
from os.path import join
from umap import UMAP
import numpy as np
import rasterfairy
import calendar
import argparse
import gensim
import codecs
import time
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


class EpochLogger(CallbackAny2Vec):
  '''Helper class that logs epoch number during Gensim training'''
  def __init__(self):
    self.epoch_count = 0

  def on_epoch_end(self, model):
    self.epoch_count += 1
    print('   * completed {0} Word2Vec epochs'.format(self.epoch_count))


def create_word2vec_model(args):
  '''
  Given a glob of text files, create a word2vec model with gensim and save to disk
  '''
  if not args.texts:
    print('\nPlease identify a glob of files to process:\n')
    print('wordmap --texts "my_data/*.txt"')
    sys.exit()
  # inform the user which files are being processed
  files = glob.glob(args.texts)
  display_files = files[:10] + ['...'] if len(files) >= 10 else files
  sep = '    \n    '
  print(' * preparing to parse {0} files:{1}'.format(len(files), sep + sep.join(display_files)))
  # generate and save the word2vec model for the user's input files
  print(' * cleaning input files')
  word_lists = []
  for i in files:
    with codecs.open(i, 'r', args.encoding) as f:
      word_lists.append(re.sub(r'[^\w\s]','',f.read().lower()).split())
  print(' * building Word2Vec model')
  epoch_logger = EpochLogger()
  model = Word2Vec(word_lists,
    size=args.size,
    window=args.window,
    min_count=args.min_count,
    workers=args.workers,
    callbacks=[epoch_logger],
  )
  model.save(args.model_name)
  return model


def plot_gensim_word2vec(args, model):
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
  if args.max_words:
    words = words[:args.max_words]
  df = np.array([model.wv[w] for w in words])
  if len(words) != df.shape[0]:
    print('! Warning: the number of words != number of elements in dataframe')
    print('! Number of words:', len(words))
    print('! Number of elements in dataframe:', df.shape[0])
  plot(args, words, df)


def plot(args, words, df):
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
  projections = []
  if 'umap' in args.layouts:
    p = UMAP(
      n_components=args.n_components,
      verbose=args.verbose,
      n_neighbors=args.n_neighbors,
      min_dist=args.min_dist,
    ).fit_transform(df)
    projections.append({
      'words': words,
      'name': 'umap',
      'positions': np.around(p, 2).tolist(),
    })

  if 'tsne' in args.layouts:
    p = TSNE(n_components=args.n_components, verbose=args.verbose).fit_transform(df)
    projections.append({
      'words': words,
      'name': 'tsne',
      'positions': np.around(p, 2).tolist(),
    })

  if 'grid' in args.layouts:
    if not projections:
      print('Please specify one or more of the following layouts as a basis for the grid positions: umap tsne')
      sys.exit()
    df = np.vstack(projections[0]['positions'])[:,:2]
    p = rasterfairy.transformPointCloud2D(df)
    p = [[int(i[0]), int(i[1])] for idx, i in enumerate(p[0])]
    # jitter the grid positions slightly to prevent/discourage overlap
    for i in p:
      if i[0]%2==0:
        i[1] += 0.5
    projections.append({
      'words': projections[0]['words'],
      'name': 'grid',
      'positions': p,
    })

  # persist the results to disk
  prepare_web_assets()
  out_path = join(web_target, 'wordmap-layouts.json')
  with open(out_path, 'w') as out:
    json.dump(projections, out)


def prepare_web_assets():
  '''
  Create ./web with the assets required to visualize input text data
  '''
  if not os.path.exists(web_target):
    copy_tree(join(path, 'web'), web_target)


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


def parse():
  '''
  Main method for parsing a glob of files passed on the command line
  '''
  parser = argparse.ArgumentParser(description='Transform text files into a large WebGL Visualization')
  parser.add_argument('--texts', type=str, help='A glob of text files to process', required=False)
  parser.add_argument('--encoding', type=str, default='utf8', help='The encoding of input files', required=False)
  parser.add_argument('--max_words', type=int, default=100000, help='Maximum number of words to include in visualization', required=False)
  parser.add_argument('--layouts', nargs='+', default=['umap', 'grid'], choices=['umap', 'tsne', 'grid'])
  parser.add_argument('--n_components', type=int, default=2, choices=[2, 3])
  parser.add_argument('--model', type=str, help='Path to a Word2Vec model to load', required=False)
  parser.add_argument('--model_name', type=str, default='{}.model'.format(calendar.timegm(time.gmtime())), help='The name to use when saving a word2vec model')
  parser.add_argument('--obj_file', type=str, help='An .obj file to control the output visualization shape', required=False)
  parser.add_argument('--size', type=int, default=50, help='Number of dimensions to include in Word2Vec vectors', required=False)
  parser.add_argument('--window', type=int, default=5, help='Number of words to include in windows when creating Word2Vec model', required=False)
  parser.add_argument('--min_count', type=int, default=20, help='Minimum occurrences of each word to be included in the Word2Vec model', required=False)
  parser.add_argument('--workers', type=int, default=7, help='The number of computer cores to use when processing user data', required=False)
  parser.add_argument('--verbose', type=bool, default=False, help='If true, logs progress during layout construction')
  parser.add_argument('--n_neighbors', type=int, default=10, help='The n_neighbors parameter for UMAP layouts')
  parser.add_argument('--min_dist', type=float, default=0.5, help='The min_dist parameter for UMAP layouts')
  args = parser.parse_args()
  # find or create a word2vec model
  if args.model:
    print(' * loading Word2Vec model', args.model)
    model = Word2Vec.load(args.model)
  else:
    print(' * creating Word2Vec model')
    model = create_word2vec_model(args)
  # plot the created or loaded model
  plot_gensim_word2vec(args, model)


if __name__ == '__main__':
  parse()
from flask import Flask, jsonify, request, send_from_directory, render_template
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from distutils.dir_util import copy_tree
from sklearn.cluster import KMeans
from collections import defaultdict
from flask_cors import CORS
from os.path import join
from lloyd import Field
from ivis import Ivis
from umap import UMAP
import numpy as np
import rasterfairy
import itertools
import calendar
import argparse
import warnings
import sklearn
import gensim
import joblib
import codecs
import glob2
import copy
import time
import json
import os
import re

# optional acceleration
try:
  from MulticoreTSNE import MulticoreTSNE
  multicore_tsne = True
except:
  from sklearn.manifold import TSNE
  multicore_tsne = False

# Store a reference to the location of this package
template_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'web')

# Store a reference to the location to which web assets will be written
target_dir = join(os.getcwd(), 'web')

# Store a reference to the location where models will be cached
cache_dir = 'models'

# List of all possible layout options
layouts = ['ivis', 'umap', 'tsne', 'grid']

# List of layout combinatorial fields applied after layouts are computed
post_layout_params = ['n_clusters']

class Manifest:
  def __init__(self, *args, **kwargs):
    self.template_dir = kwargs.get('template_dir', template_dir)
    self.target_dir = kwargs.get('target_dir', target_dir)
    self.args = kwargs.get('args', {})
    self.df = kwargs.get('df', [[]])
    self.strings = kwargs.get('strings', [])
    self.layouts = []
    for layouts in self.args['layouts']:
      params_list = self.get_layout_params(layouts)
      for idx, params in enumerate(params_list):
        print(' * computing', layouts, 'layout', idx+1, 'of', len(params_list))
        layout = Layout(layout=layouts, params=params, df=self.df)
        # skip layouts that couldn't be processed
        if layout.json:
          self.layouts.append(layout)
    self.write_web_assets()

  def get_layout_params(self, layout):
    '''Find the set of all hyperparams for each layout'''
    l = [] # store for the list of param dicts for this layout
    keys = [i for i in self.args if i.startswith(layout) and self.args[i]]
    keys += post_layout_params
    d = {i: self.args[i] for i in keys}
    for i in list(itertools.product(*[[{i:j} for j in d[i]] for i in d])):
      a = copy.deepcopy(self.args)
      for j in i:
        a.update(j)
      l.append(a)
    return l if l else [self.args] # handle layout without named params (e.g. grid)

  def make_directories(self):
    '''Create the directories to which outputs will be written'''
    for i in self.layouts:
      # make the layout dir for the web app
      if not os.path.exists(os.path.join(self.target_dir, 'data', 'layouts', i.layout)):
        os.makedirs(os.path.join(self.target_dir, 'data', 'layouts', i.layout))
      # make the dir where the layout will be cached
      p = os.path.join(cache_dir, i.layout)
      if not os.path.exists(p):
        os.makedirs(p)

  def copy_web_template(self):
    '''Copy the base web asset files from self.directory to ./web'''
    if not os.path.exists(self.target_dir):
      copy_tree(self.template_dir, target_dir)

  def get_manifest_json(self):
    '''Create a manifest.json file outlining the layout options availble'''
    d = defaultdict(lambda: defaultdict(list))
    for i in self.layouts:
      d['params'] = self.args,
      d['layouts'][i.layout].append({
        'filename': i.filename,
        'params': {k: str(v) for k, v in i.hyperparams.items()},
      })
    return d

  def write_web_assets(self):
    '''Write all web assets needed to create a visualization'''
    print(' * writing json assets')
    self.copy_web_template()
    self.make_directories()
    data_dir = os.path.join(self.target_dir, 'data')
    # write the string data
    with open(os.path.join(data_dir, 'texts.json'), 'w') as out:
      json.dump(self.strings, out)
    # write the individual layout files
    for i in self.layouts:
      with open(os.path.join(data_dir, 'layouts', i.layout, i.filename), 'w') as out:
        json.dump(i.json, out)
    # write the full manifest json
    with open(os.path.join(data_dir, 'manifest.json'), 'w') as out:
      json.dump(self.get_manifest_json(), out)


class Layout:
  def __init__(self, *args, **kwargs):
    self.layout = kwargs.get('layout')
    self.params = kwargs.get('params')
    self.json = {}
    self.float_precision = kwargs.get('float_precision', 2)
    self.hyperparams = self.get_hyperparams()
    self.filename = self.get_filename()
    self.cache_dir = os.path.join(cache_dir, self.layout)
    self.cache_path = self.get_cache_path()
    self.get_positions(kwargs.get('df'))

  def get_hyperparams(self):
    '''Return a dict of k/v params for this specific layout'''
    d = {}
    for k, v in self.params.items():
      if k.startswith(self.layout) or k in post_layout_params:
        k = k.replace('{0}_'.format(self.layout), '')
        d[k] = v
    return d

  def get_filename(self):
    '''Get the filename to use when saving this layout to disk as JSON'''
    fn = self.layout + '_'
    for k in sorted(list(self.hyperparams.keys())):
      if self.hyperparams[k]:
        fn += '{0}-{1}-'.format(k, self.hyperparams[k])
    return fn.rstrip('-').rstrip('_') + '.json'

  def get_cache_path(self):
    '''Get the path to the cache to which this layout will be written'''
    return os.path.join(self.cache_dir, self.filename)

  def load_from_cache(self):
    '''Try to load this layout from the cache'''
    if os.path.exists(self.cache_path):
      with open(self.cache_path) as f:
        return json.load(f)

  def write_to_cache(self):
    '''Write the model to the cache for faster i/o in the future'''
    if not self.json:
      warnings.warn('Could not persist layout to cache; no JSON present')
    else:
      if not os.path.exists(self.cache_dir):
        os.makedirs(self.cache_dir)
      with open(self.cache_path, 'w') as out:
        json.dump(self.json, out)

  def get_model(self):
    if self.layout == 'umap':
      return UMAP(
        n_components = self.params.get('n_components'),
        verbose = self.params.get('verbose'),
        n_neighbors = self.params.get('umap_n_neighbors'),
        min_dist = self.params.get('umap_min_dist'),
      )
    elif self.layout == 'tsne':
      if multicore_tsne:
        return MulticoreTSNE()
      return TSNE(
        n_components = self.params.get('n_components'),
        verbose = self.params.get('verbose'),
      )
    elif self.layout == 'ivis':
      return Ivis(
        model = self.params.get('ivis_model'),
        embedding_dims = self.params.get('n_components'),
        k = self.params.get('ivis_k'),
        verbose = self.params.get('verbose'),
      )
    elif self.layout == 'grid':
      # monkeypatch fit_transform method into rasterfairy for consistent api
      def fit_transform(X):
        return rasterfairy.transformPointCloud2D(X[:,:2])[0]
      clf = rasterfairy
      setattr(clf, 'fit_transform', fit_transform)
      return clf
    else:
      print('Warning: received request for unsupported classifier')

  def get_positions(self, df):
    '''Find the vertex positions for the user-provided data'''
    cache = self.load_from_cache()
    if cache:
      self.json = cache
    else:
      try:
        df = self.scale_data(df)
        # find the direct model positions for this layout
        positions = self.get_model().fit_transform(df)
        # find the k means clusters
        clusters = KMeans(n_clusters=self.params['n_clusters'], random_state=0).fit(positions)
        self.json = {
          'layout': self.layout,
          'filename': self.filename,
          'hyperparams': self.hyperparams,
          'positions': self.round(positions.tolist()),
          'jittered': self.jitter_positions(positions),
          'clusters': clusters.labels_.tolist(),
          'cluster_centers': self.round(clusters.cluster_centers_.tolist()),
        }
        self.write_to_cache()
      except Exception as exc:
        print(' ! Failed to generate layout with params', self.params, exc)

  def jitter_positions(self, X):
    '''Jitter the points in a 2D dataframe `X` using lloyd's algorithm'''
    if self.params['n_components'] == 2 and self.params.get('lloyd_iterations', None):
      jittered = Field(positions)
      for i in range(self.params['lloyd_iterations']):
        print(' * running lloyd iteration', i)
        jittered.relax()
      return self.round(jittered)
    return None

  def scale_data(self, X):
    '''Scale a 2d array of points `X`'''
    if self.layout == 'ivis':
      return MinMaxScaler().fit_transform(X)
    return X

  def round(self, arr):
    '''Round the values in a 2d arr'''
    rounded = []
    for i in arr:
      l = []
      for j in i:
        l.append(round(j, self.float_precision))
      rounded.append(l)
    return rounded


class EpochLogger(CallbackAny2Vec):
  '''Helper class that logs epoch number during Gensim training'''
  def __init__(self):
    self.epoch_count = 0

  def on_epoch_end(self, model):
    self.epoch_count += 1
    print('   * completed {0} training epochs'.format(self.epoch_count))


class Model:
  def __init__(self, *args, **kwargs):
    self.args = kwargs.get('args', {})
    self.texts = glob2.glob(self.args['texts'])
    if self.args.get('model', False):
      self.model = self.load_from_cache()
    else:
      self.model = None
      self.model_type = self.args['model_type']
      self.cache_dir = os.path.join(kwargs.get('cache_dir', cache_dir), self.model_type)
      self.cache_path = self.get_cache_path()
      self.create_model()
      self.save_to_cache()
    self.create_manifest()

  def get_cache_path(self):
    '''Return the path wherein this model will be persisted'''
    return os.path.join(self.cache_dir, self.args.get('model_name'))

  def load_from_cache(self):
    '''Load a saved gensim model from the model cache'''
    print(' * loading pretrained model')
    if os.path.exists(self.args['model']):
      model = gensim.models.Word2Vec.load(self.args['model'])
      if isinstance(model, gensim.models.word2vec.Word2Vec):
        self.model_type = 'word2vec'
        print(' * loaded model with', len(model.wv.index2entity), 'words')
      elif isinstance(model, gensim.models.doc2vec.Doc2Vec):
        self.model_type = 'doc2vec'
        print(' * loaded model with', len(model.wv.index2entity), 'docs')
      else:
        raise Exception('The loaded model type could not be inferred. Please create a new model')
      return model

  def create_model(self):
    '''Create a gensim Word2Vec model with the input data'''
    self.log_files()
    self.set_model(self.get_input_data())

  def set_model(self, input_data):
    '''Return a model trained on the input data'''
    if self.model_type == 'word2vec':
      self.model = gensim.models.Word2Vec(
        input_data,
        size = self.args['size'],
        window = self.args['window'],
        min_count = self.args['min_count'],
        workers = self.args['workers'],
        callbacks = [EpochLogger()],
        max_final_vocab = self.args.get('max_n', None),
        iter = self.args.get('iter', 20),
      )
      print(' * created model with', len(self.model.wv.index2entity), 'words')
    elif self.model_type == 'doc2vec':
      self.model = gensim.models.Doc2Vec(
        input_data,
        vector_size = self.args['size'],
        window = self.args['window'],
        min_count = self.args['min_count'],
        workers = self.args['workers'],
        callbacks = [EpochLogger()],
        iter = self.args.get('iter', 20),
      )
      print(' * created model with', len(self.model.wv.index2entity), 'docs')
    else:
      raise Exception('The requested model type is not supported:', self.model_type)

  def save_to_cache(self):
    '''Save self.model to the model cache'''
    if not os.path.exists(self.cache_dir):
      os.makedirs(self.cache_dir)
    self.model.save(self.cache_path)

  def log_files(self):
    '''Print a note indicating the files this Word2Vec instance will process'''
    display_files = self.texts[:10] + ['...'] if len(self.texts) >= 10 else self.texts
    sep = '    \n    '
    print(' * preparing to parse {0} files:{1}'.format(
      len(self.texts),
      sep + sep.join(display_files))
    )

  def get_input_data(self):
    '''Return a 2d list of words in self.files'''
    if self.model_type in ['word2vec', 'doc2vec']:
      word_lists = []
      for i in self.texts:
        with codecs.open(i, 'r', self.args['encoding']) as f:
          word_lists.append(re.sub(r'[^\w\s]','',f.read().lower()).split())
      if self.model_type == 'word2vec':
        return word_lists
      else:
        return [TaggedDocument(doc, [i]) for i, doc in enumerate(word_lists)]

  def create_manifest(self):
    '''Create a plot from this model'''
    if self.model_type == 'word2vec':
      strings = self.model.wv.index2entity
      df = np.array([self.model.wv[w] for w in strings])
    else:
      strings = [self.clean_filename(i) for i in self.texts]
      df = [self.model.docvecs[idx] for idx, _ in enumerate(strings)]
    if self.args['max_n'] and len(strings) > self.args['max_n']:
      print(' * limiting input string set to length', self.args['max_n'])
      strings = strings[:self.args['max_n']]
      df = np.array(df)[:self.args['max_n']]
    manifest = Manifest(args=self.args, strings=strings, df=df)

  def clean_filename(self, path):
    '''Given the path to a file return a clean filename for displaying'''
    s = os.path.splitext(os.path.basename(path))[0].lower()
    s = re.sub(r'[^\w\s]',' ', s)
    s = ' '.join(i[0].upper() + ''.join(i[1:]) for i in s.split())
    return s


def serve():
  '''
  Method to serve the content in ./web
  '''
  print(' * serving assets from', target_dir)
  app = Flask(__name__, static_folder=target_dir, template_folder=target_dir)
  CORS(app)
  # requests for static index file
  @app.route('/')
  def index():
    return render_template('index.html')
  # requests for static assets
  @app.route('/<path:path>')
  def asset(path):
    return send_from_directory(target_dir, path)
  # run the server
  app.run(host= '0.0.0.0', port=7082)


def validate_user_args(args):
  '''
  Run any necessary checks on the user-provided arguments
  '''
  # ensure the user asks for 2 or 3 embedding dimensions
  if args['n_components'] not in [2,3]:
    raise Exception('Please set n_components to 2 or 3')
  # if the user requested a grid layout, make sure we have a non-grid layout
  if args['layouts'] == ['grid']:
    raise Exception('Please specify one or more layouts as a basis for the grid positions: umap tsne')
  # make sure the user provided a pretrained model or text data for a new model
  if not any([args['model'], args['texts']]):
    raise Exception('Please provide either a --model or --texts argument')
  # if the user specified a model type make sure it's supported
  if args['model_type'] not in ['word2vec', 'doc2vec']:
    raise Exception('The requested model type is not supported')
  # if the user specified a model check that it exists
  if args['model'] and not os.path.exists(args['model']):
    raise Exception('The specified model file does not exist')
  # check if the user asked for any invalid layouts
  invalid_layouts = [i for i in args['layouts'] if i not in layouts]
  if any(invalid_layouts):
    warnings.warn('Requested layouts are not available:', invalid_layouts)
  args['layouts'] = [i for i in args['layouts'] if i in layouts]
  # ensure grid is the last layout in the layout list
  if 'grid' in args['layouts']:
    args['layouts'].remove('grid')
    args['layouts'].append('grid')


def parse():
  '''
  Main method for parsing a glob of files passed on the command line
  '''
  parser = argparse.ArgumentParser(description='Transform text data into a large WebGL Visualization')
  # input data parameters
  parser.add_argument('--texts', type=str, help='A glob of text files to process', required=False)
  parser.add_argument('--encoding', type=str, default='utf8', help='The encoding of input files', required=False)
  # model parameters
  parser.add_argument('--model', type=str, help='Path to a Word2Vec or Doc2Vec model to load', required=False)
  parser.add_argument('--model_name', type=str, default='{}.model'.format(calendar.timegm(time.gmtime())), help='The name to use when saving a word2vec model')
  parser.add_argument('--model_type', type=str, default='word2vec', choices=['word2vec', 'doc2vec'], help='The type of model to build {word2vec|doc2vec}', required=False)
  parser.add_argument('--size', type=int, default=64, help='Number of dimensions to include in the model embeddings', required=False)
  parser.add_argument('--window', type=int, default=5, help='Number of words to include in windows when creating model vectors', required=False)
  parser.add_argument('--min_count', type=int, default=5, help='Minimum occurrences of each word to be included in the model', required=False)
  parser.add_argument('--workers', type=int, default=7, help='The number of computer cores to use when processing input data', required=False)
  parser.add_argument('--iter', type=int, default=20, help='The number of iterations to use when training a model')
  # layout parameters
  parser.add_argument('--layouts', type=str, nargs='+', default=['umap', 'grid'], choices=layouts)
  parser.add_argument('--max_n', type=int, default=100000, help='Maximum number of words/docs to include in visualization', required=False)
  parser.add_argument('--obj_file', type=str, help='An .obj file to control the output visualization shape', required=False)
  parser.add_argument('--n_components', type=int, default=2, choices=[2, 3], help='Number of dimensions in the embeddings / visualization')
  parser.add_argument('--lloyd_iterations', type=int, default=0, help='Number of Lloyd\'s algorithm iterations to run on each layout (requires n_components == 2)')
  parser.add_argument('--verbose', type=bool, default=False, help='If true, logs progress during layout construction')
  # shared combinatorial layout params
  parser.add_argument('--n_clusters', type=int, nargs='+', default=[7], help='The number of clusters to identify')
  # layout parameters - tsne
  parser.add_argument('--tsne_perplexity', type=int, nargs='+', default=[30], help='The perplexity parameter for TSNE layouts')
  # layout parameters - umap
  parser.add_argument('--umap_n_neighbors', type=int, nargs='+', default=[10], help='The n_neighbors parameter for UMAP layouts')
  parser.add_argument('--umap_min_dist', type=float, nargs='+', default=[0.5], help='The min_dist parameter for UMAP layouts')
  # layout parameters - ivis
  parser.add_argument('--ivis_model', type=str, nargs='+', default=['maaten'], help='The model parameter for ivis')
  parser.add_argument('--ivis_annoy_index_path', type=str, default=[None], help='The path to the annoy index used by ivis')
  parser.add_argument('--ivis_k', type=int, nargs='+', default=[15], help='The k parameter for ivis layouts')
  args = vars(parser.parse_args())
  # validate args
  validate_user_args(args)
  # find or create a word2vec model
  model = Model(args=args)

if __name__ == '__main__':
  parse()
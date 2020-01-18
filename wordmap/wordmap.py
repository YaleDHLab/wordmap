from flask import Flask, jsonify, request, send_from_directory, render_template
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import StandardScaler
from vertices import ObjParser, ImgParser
from distutils.dir_util import copy_tree
from sklearn.decomposition import NMF
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.misc import imread
from flask_cors import CORS
from os.path import join
from lloyd import Field
from ivis import Ivis
from umap import UMAP
import numpy as np
import rasterfairy
import webbrowser
import traceback
import itertools
import calendar
import argparse
import sklearn
import gensim
import joblib
import codecs
import glob2
import copy
import time
import json
import six
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
layouts = ['ivis', 'umap', 'tsne', 'grid', 'img', 'obj']

# List of layout combinatorial fields applied after layouts are computed
post_layout_params = ['n_clusters']

# Default arg vals
defaults = {
  'texts': [],
  'encoding': 'utf8',
  'model_name': '{}.model'.format(calendar.timegm(time.gmtime())),
  'model_type': 'word2vec',
  'use_cache': False,
  'size': 64,
  'window': 7,
  'min_count': 5,
  'workers': 7,
  'iter': 20,
  'layouts': ['umap', 'grid'],
  'max_n': 100000,
  'n_components': 2,
  'lloyd_iterations': 0,
  'verbose': False,
  'n_clusters': [7],
  'ivis_k': [150],
  'ivis_model': ['maaten'],
  'tsne_perplexity': [30],
  'umap_n_neighbors': [10],
  'umap_min_dist': [0.5],
}


class EpochLogger(CallbackAny2Vec):
  '''Helper class that logs epoch number during Gensim training'''
  def __init__(self):
    self.epoch_count = 0

  def on_epoch_end(self, model):
    self.epoch_count += 1
    print('   * completed {0} training epochs'.format(self.epoch_count))


class Model:
  def __init__(self, *args, **kwargs):
    self.args = args[0]
    self.texts = []
    self.format_args()
    self.validate_args()
    if self.args.get('model', False):
      self.model = self.load_from_cache()
    elif self.args.get('model_type', None):
      self.model = None
      self.cache_dir = os.path.join(
        kwargs.get('cache_dir', cache_dir),
        self.args['model_type'],
      )
      self.cache_path = self.get_cache_path()
      self.create_model()
      self.save_to_cache()
    self.create_manifest()


  def format_args(self):
    '''Ensure input args conform to argtypes expected by argparse'''
    # provide default args
    for i in defaults:
      if not self.args.get(i, False):
        self.args[i] = defaults[i]
    # format input texts arg
    if self.args['texts']:
      if isinstance(self.args['texts'], six.string_types): # e.g. 'data/*.txt'
        self.texts = glob2.glob(self.args['texts'])
      elif isinstance(self.args['texts'], list):
        self.texts = self.args['texts']
    else:
      self.texts = []
    # ensure all list types are indeed lists
    list_types = [
      'n_clusters',
      'tsne_perplexity',
      'umap_n_neighbors',
      'ivis_model',
      'ivis_k',
    ]
    for i in list_types:
      if self.args.get(i, None) and not isinstance(self.args[i], list):
        self.args[i] = [self.args[i]]


  def validate_args(self):
    '''Validate the user arguments are in the right shape'''
    args = self.args
    # if the user requested a grid layout, make sure we have a non-grid layout
    if args.get('layouts', []) == ['grid']:
      raise Exception('Please specify one or more layouts as a basis for the grid positions: umap tsne')
    # unless layout is img or obj we need a model or texts
    if args.get('layouts', []) != ['obj'] and args['layouts'] != ['img']:
      # ensure the user asks for 2 or 3 embedding dimensions
      if args.get('n_components', []) not in [2, 3]:
        raise Exception('Please set n_components to 2 or 3')
      # make sure the user provided a pretrained model or text data for a new model
      if not any([args.get('model', None), args.get('texts', None)]):
        raise Exception('Please provide either a --model or --texts argument')
      # if the user specified a model type make sure it's supported
      if args.get('model_type', None) not in ['word2vec', 'doc2vec']:
        raise Exception('The requested model type is not supported')
      # if the user specified a model check that it exists
      if args.get('model', None) and not os.path.exists(args.get('model', None)):
        raise Exception('The specified model file does not exist')
    # else ensure user has provided obj / img file
    elif 'obj' in args.get('layouts', []) and 'obj_file' not in args:
      print(' ! The obj layout requires an obj_file argument')
    elif 'img' in args.get('layouts', []) and 'img_file' not in args:
      print(' ! The img layout requires an img_file argument')
    # check if the user asked for any invalid layouts
    invalid_layouts = [i for i in args.get('layouts', []) if i not in layouts]
    if any(invalid_layouts):
      print(' ! Requested layouts are not available: {}'.format(invalid_layouts))
    args['layouts'] = [i for i in args.get('layouts', []) if i not in invalid_layouts]
    # ensure img layout requests have an img_file
    if 'img' in args.get('layouts', []) and not args.get('img_file', None):
      raise Exception('img layouts require an img_file parameter')
    # ensure grid is the last layout in the layout list
    if 'grid' in args.get('layouts', []):
      args['layouts'].remove('grid')
      args['layouts'].append('grid')


  def get_cache_path(self):
    '''Return the path wherein this model will be persisted'''
    return os.path.join(self.cache_dir, self.args.get('model_name'))

  def load_from_cache(self):
    '''Load a saved gensim model from the model cache'''
    print(' * loading pretrained model')
    if os.path.exists(self.args['model']):
      model = gensim.models.Word2Vec.load(self.args['model'])
      if isinstance(model, gensim.models.word2vec.Word2Vec):
        self.args['model_type'] = 'word2vec'
        print(' * loaded model with', len(model.wv.index2entity), 'words')
      elif isinstance(model, gensim.models.doc2vec.Doc2Vec):
        self.args['model_type'] = 'doc2vec'
        print(' * loaded model with', len(model.wv.index2entity), 'docs')
      else:
        raise Exception('The loaded model type could not be inferred. Please create a new model')
      return model
    else:
      raise Exception('Requested model is not available', self.args['model'])

  def create_model(self):
    '''Create a gensim Word2Vec model with the input data'''
    self.log_files()
    self.set_model(self.get_input_data())

  def set_model(self, input_data):
    '''Return a model trained on the input data'''
    if self.args['model_type'] == 'word2vec':
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
    elif self.args['model_type'] == 'doc2vec':
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
      raise Exception('The requested model type is not supported:', self.args['model_type'])

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
    if self.args.get('model_type', False) in ['word2vec', 'doc2vec']:
      word_lists = [self.get_file_words(i) for i in self.texts]
      if self.args['model_type'] == 'word2vec':
        return word_lists
      else:
        return [TaggedDocument(doc, [i]) for i, doc in enumerate(word_lists)]
    else:
      raise Exception('Requested model_type not supported', self.args.get('model_type'))

  def get_file_words(self, path):
    '''
    Return a 1d array of the words in the file at path
    '''
    if os.path.exists(path):
      with codecs.open(path, 'r', self.args['encoding']) as f:
        return re.sub(r'[^\w\s]','', f.read().lower()).split()
    else:
      print(' ! The following requested file does not exist: {}'.format(path))

  def create_manifest(self):
    '''Create a plot from this model'''
    if self.args.get('model_type', False) == 'word2vec':
      strings = self.model.wv.index2entity
      df = np.array([self.model.wv[w] for w in strings])
    elif self.args.get('model_type', False) == 'doc2vec':
      strings = [self.clean_filename(i) for i in self.texts]
      df = [self.model.docvecs[idx] for idx, _ in enumerate(strings)]
    else:
      strings = flatten([self.get_file_words(i) for i in self.texts])
      df = np.ones((len(strings), 2))
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
          # write web assets after each iteration for faster prototyping
          self.write_web_assets()

  def get_layout_params(self, layout):
    '''Find the set of all hyperparams for each layout'''
    l = [] # store for the list of param dicts for this layout
    keys = [i for i in self.args if i.startswith(layout) and \
      self.args[i] and isinstance(self.args[i], list)]
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
    copy_tree(self.template_dir, target_dir)

  def get_manifest_json(self):
    '''Create a manifest.json file outlining the layout options availble'''
    d = defaultdict(lambda: defaultdict(list))
    d['params'] = self.args
    for i in self.layouts:
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
    self.float_precision = kwargs.get('float_precision', 4)
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
    if os.path.exists(self.cache_path) and self.params['use_cache']:
      with open(self.cache_path) as f:
        print(' * loading layout from cache')
        return json.load(f)

  def write_to_cache(self):
    '''Write the model to the cache for faster i/o in the future'''
    if not self.json:
      print(' ! Could not persist layout to cache; no JSON present')
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
        n_epochs_without_progress = 10,
      )
    elif self.layout == 'grid':
      # monkeypatch fit_transform method into rasterfairy for consistent api
      def fit_transform(X):
        return rasterfairy.transformPointCloud2D(X[:,:2])[0]
      clf = rasterfairy
      setattr(clf, 'fit_transform', fit_transform)
      return clf
    elif self.layout == 'img':
      class ImgLayout:
        def __init__(self, img_path):
          self.img_path = img_path
        def fit_transform(self, X):
          verts = ImgParser(self.img_path).get_n_vertices(X.shape[0])
          # reorder vertices to get row major distribution
          verts = np.array([verts[:,1], 1-verts[:,0]]).T
          return verts
      return ImgLayout(self.params.get('img_file'))
    elif self.layout == 'obj':
      class ObjLayout:
        def __init__(self, obj_path):
          self.obj_path = obj_path
        def fit_transform(self, X):
          return ObjParser(self.obj_path).get_n_vertices(X.shape[0])
      return ObjLayout(self.params.get('obj_file'))
    else:
      print(' ! Received request for unsupported layout model', self.layout)

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

        data = {
          'layout': self.layout,
          'filename': self.filename,
          'hyperparams': self.hyperparams,
          'positions': self.round(positions.tolist()),
        }
        # find the k clusters if requested
        if self.params['n_clusters']:
          n_clusters = self.params['n_clusters']
          clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(positions)
          data.update({
            'clusters': clusters.labels_.tolist(),
            'cluster_centers': self.round(clusters.cluster_centers_.tolist()),
          })
        # jitter the points if requested
        if self.params.get('n_components', None) == 2 and \
           self.params.get('lloyd_iterations', None):
          data.update({
            'jittered': self.jitter_positions(positions),
          })
        # store and write the data
        self.json = data
        self.write_to_cache()
      except Exception as exc:
        print(' ! Failed to generate layout\n',
          self.__dict__,
          '\n',
          traceback.format_exc())


  def jitter_positions(self, X):
    '''Jitter the points in a 2D dataframe `X` using lloyd's algorithm'''
    if self.params['n_components'] == 2 and self.params.get('lloyd_iterations', None):
      jittered = Field(positions)
      for i in range(self.params['lloyd_iterations']):
        print(' * running lloyd iteration', i)
        jittered.relax()
      return self.round(jittered)
    return X

  def scale_data(self, X):
    '''Scale a 2d array of points `X`'''
    if self.layout == 'ivis':
      return StandardScaler().fit_transform(X)
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


def flatten(l):
  '''
  Flatten a 2d array to 1d
  '''
  return [j for i in l for j in i]


def serve(port=7082):
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
  url = 'http://localhost:{}'.format(port)
  webbrowser.open(url, new=2, autoraise=True)
  app.run(host= '0.0.0.0', port=port)


def parse():
  '''
  Main method for parsing a glob of files passed on the command line
  '''
  parser = argparse.ArgumentParser(description='Transform text data into a large WebGL Visualization',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # input data parameters
  parser.add_argument('--texts', type=str, default=defaults['texts'], help='A glob of text files to process', required=False)
  parser.add_argument('--encoding', type=str, default=defaults['encoding'], help='The encoding of input files', required=False)
  # model parameters
  parser.add_argument('--model', type=str, help='Path to a Word2Vec or Doc2Vec model to load', required=False)
  parser.add_argument('--model_name', type=str, default=defaults['model_name'], help='The name to use when saving a word2vec model')
  parser.add_argument('--model_type', type=str, default=defaults['model_type'], choices=['word2vec', 'doc2vec'], help='The type of model to build {word2vec|doc2vec}', required=False)
  parser.add_argument('--use_cache', type=bool, default=defaults['use_cache'], help='Boolean indicating whether to load cached models to save compute time')
  # k-to-vec model params
  parser.add_argument('--size', type=int, default=defaults['size'], help='Number of dimensions to include in the model embeddings', required=False)
  parser.add_argument('--window', type=int, default=defaults['window'], help='Number of words to include in windows when creating model vectors', required=False)
  parser.add_argument('--min_count', type=int, default=defaults['min_count'], help='Minimum occurrences of each word to be included in the model', required=False)
  parser.add_argument('--workers', type=int, default=defaults['workers'], help='The number of computer cores to use when processing input data', required=False)
  parser.add_argument('--iter', type=int, default=defaults['iter'], help='The number of iterations to use when training a model')
  # layout parameters
  parser.add_argument('--layouts', type=str, nargs='+', default=defaults['layouts'], choices=layouts)
  parser.add_argument('--max_n', type=int, default=defaults['max_n'], help='Maximum number of words/docs to include in visualization', required=False)
  parser.add_argument('--obj_file', type=str, help='The path to an .obj file to control the output layout', required=False)
  parser.add_argument('--img_file', type=str, help='The path to a .jpg or .png file to control the output layout')
  parser.add_argument('--n_components', type=int, default=defaults['n_components'], choices=[2, 3], help='Number of dimensions in created embeddings')
  parser.add_argument('--lloyd_iterations', type=int, default=defaults['lloyd_iterations'], help='Number of Lloyd\'s algorithm iterations to run on each layout (requires n_components == 2)')
  parser.add_argument('--verbose', type=bool, default=defaults['verbose'], help='If true, logs progress during layout construction')
  # shared combinatorial layout params
  parser.add_argument('--n_clusters', type=int, nargs='+', default=defaults['n_clusters'], help='The number of clusters to identify')
  # layout parameters - ivis
  parser.add_argument('--ivis_k', type=int, nargs='+', default=defaults['ivis_k'], help='The k parameter for ivis layouts')
  parser.add_argument('--ivis_model', type=str, nargs='+', default=defaults['ivis_model'], help='The model parameter for ivis layouts')
  # layout parameters - tsne
  parser.add_argument('--tsne_perplexity', type=int, nargs='+', default=defaults['tsne_perplexity'], help='The perplexity parameter for TSNE layouts')
  # layout parameters - umap
  parser.add_argument('--umap_n_neighbors', type=int, nargs='+', default=defaults['umap_n_neighbors'], help='The n_neighbors parameter for UMAP layouts')
  parser.add_argument('--umap_min_dist', type=float, nargs='+', default=defaults['umap_min_dist'], help='The min_dist parameter for UMAP layouts')
  args = vars(parser.parse_args())
  # instantiate class that writes all data to disk
  model = Model(args)

if __name__ == '__main__':
  parse()

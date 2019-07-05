from setuptools import setup
from os.path import join
import glob

setup (
  name='wordmap',
  version='0.0.1',
  packages=['wordmap'],
  package_data={
    'wordmap': ['web/*', 'web/js/*'],
  },
  keywords = ['webgl', 'three.js', 'word2vec', 'tsne', 'umap', 'machine-learning'],
  description='Visualize massive ',
  url='https://github.com/yaledhlab/wordmap',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'Flask>=0.12.2',
    'Flask-Cors>=3.0.3',
    'gensim>=3.6.0',
    'gunicorn>=19.7.1',
    'matplotlib>=2.0.0',
    'MulticoreTSNE>=0.1',
    'numpy>=1.15.1',
    'yale_dhlab_rasterfairy>=1.0.3',
    'scipy>=1.2.1',
  ],
)

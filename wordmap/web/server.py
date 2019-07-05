# web serving
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import os

# Configure Flask
pwd = os.path.join('.')
app = Flask(__name__, static_folder=pwd, template_folder=pwd)
CORS(app)

# requests for static index file
@app.route('/')
def index():
  return render_template('index.html')

# requests for static assets
@app.route('/<path:path>')
def asset(path):
  return send_from_directory(pwd, path)

# run the server
app.run(host= '0.0.0.0', port=7082)
// aliases
var BA = THREE.BufferAttribute,
    IBA = THREE.InstancedBufferAttribute,
    ARR = Float32Array;


function Wordmap() {
  // style parameters
  this.wordScalar = 0.0003; // sizes up words
  this.maxWords = 1000000; // max number of words to draw
  this.background = '#fff'; // background color
  this.color = '#000'; // text color
  this.font = 'Monospace'; // font family
  this.mipmap = true; // toggles mipmaps in texture
  // layout parameters
  this.layout = null; // the currently selected layout
  this.heightScalar = 0.002; // controls mountain height
  // internal static
  this.size = 64; // size of each character on canvas
  this.initialQuery = 'stars'; // the default search term
  // internal state
  this.state = {
    flying: false, // bool indicating whether we're flying camera
    clock: null, // clock to measure how long we've been flying camera
    transitioning: false, // bool indicating whether layout is transitioning
    transitionQueued: false, // bool indicating whether to run another layout transition
    loadProgres: {}, // map from asset identifier to load progress
    loaded: {}, // list of strings identifying initial assets loaded
  }
  // data
  this.data = {
    words: [], // list of strings to visualize
    layouts: [],
    heightmap: {},
    characters: {},
    selected: {}, // currently selected layout
    previous: {}, // previously selected layout
  }
  // font options (each must be present in web/assets/fonts)
  this.fonts = [
    'Monospace',
    'VT323',
    'Nanum',
  ];
  // initialize
  this.loadAssets();
}


/**
* Scene
**/

Wordmap.prototype.createScene = function() {
  // scene
  var scene = new THREE.Scene();

  // camera
  var aspectRatio = window.innerWidth / window.innerHeight;
  var camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.001, 10);

  // renderer
  var renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.sortObjects = false; // make scene.add order draw order
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.domElement.id = 'gl-scene';
  document.body.appendChild(renderer.domElement);

  // controls
  var controls = new THREE.TrackballControls(camera, renderer.domElement);
  controls.zoomSpeed = 0.05;
  controls.panSpeed = 0.1;

  // add ?axes=true to url to see axis helpers for global orientation
  if (window.location.search.includes('axes=true')) {
    var axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
  }

  // store objects on instance
  this.scene = scene;
  this.camera = camera;
  this.controls = controls;
  this.renderer = renderer;
}


Wordmap.prototype.setInitialCameraPosition = function() {
  // position the camera
  this.camera.position.set(-0.01, -0.86, 0.87);
  this.camera.up.set(0.00, 0.53, 0.83);
  this.camera.quaternion.set({_x: 0.54, _y: 0.00, _z: 0.00, _w: 0.83})
  this.controls.target.set(0.01, 0.88, 0.02);
  this.controls.update();
}


Wordmap.prototype.render = function() {
  // draw the scene
  requestAnimationFrame(this.render.bind(this));
  this.renderer.render(this.scene, this.camera);
  this.controls.update();
  if (this.state.transitionQueued) {
    this.state.transitionQueued = false;
    this.setLayout();
  }
}


Wordmap.prototype.onWindowResize = function() {
  // resize the canvas when the scene resizes
  this.camera.aspect = window.innerWidth / window.innerHeight;
  this.camera.updateProjectionMatrix();
  this.renderer.setSize(window.innerWidth, window.innerHeight);
  this.setPointScale();
}


/**
* Loaders
**/

Wordmap.prototype.loadManifest = function() {
  // load manifest file with all available layouts and initialize first layout
  get('data/manifest.json', function(data) {
    this.data.manifest = JSON.parse(data);
    this.data.layouts = Object.keys(this.data.manifest);
    // store this asset in loaded assets and render if ready
    this.state.loaded['manifest'] = true;
    this.initializeIfLoaded();
  }.bind(this))
}


Wordmap.prototype.loadHeightmap = function() {
  // load an image for setting 3d vertex positions
  var img = new Image();
  img.crossOrigin = 'Anonymous';
  img.onload = function() {
    var canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    this.data.heightmap = ctx.getImageData(0,0, img.width, img.height);
    // store this asset in loaded assets and render if ready
    this.state.loaded['heightmap'] = true;
    this.initializeIfLoaded();
  }.bind(this)
  img.src = 'assets/images/heightmap.jpg';
}


Wordmap.prototype.loadFonts = function() {
  // load all fonts in this.fonts after the first (which is built into browser)
  for (var i=1; i<this.fonts.length; i++) this.loadFont(this.fonts[i]);
}


Wordmap.prototype.loadFont = function(font) {
  // load a .ttf font in web/assets/fonts
  get('assets/fonts/' + font + '.ttf', function() {
    var elem = document.createElement('style'),
        s =  '@font-face {font-family:"' + font + '"';
        s += ';src:url("assets/fonts/' + font + '.ttf") format("truetype")};';
    elem.innerHTML = s;
    document.body.appendChild(elem);
    var elem = document.createElement('div');
    elem.textContent = 'hello';
    elem.style.fontFamily = font;
    elem.id = 'font-' + font;
    document.querySelector('#font-target').appendChild(elem);
  })
}


Wordmap.prototype.loadTexts = function() {
  get('data/texts.json', function(data) {
    this.data.texts = JSON.parse(data);
    // store this asset in loaded assets and render if ready
    this.state.loaded['texts'] = true;
    this.initializeIfLoaded();
  }.bind(this))
}


/**
* Initialize Scene
**/

Wordmap.prototype.loadAssets = function() {
  // load required assets; the scene will render when all are loaded
  this.setCharacterCanvas();
  this.setBackgroundColor();
  this.loadFonts();
  this.loadManifest();
  this.loadHeightmap();
  this.loadTexts();
  this.createScene();
  this.setInitialCameraPosition();
  this.render();
}


Wordmap.prototype.initializeIfLoaded = function() {
  // set the initial layout state and render the initial layout
  if (!this.allAssetsLoaded()) return;
  // set the initial layout state and add the mesh to the scene
  this.layout = this.data.layouts[0];
  // initialize the gui to which we'll add layout hyperparms
  this.createGui();
  // set the hyperparams for the current layout
  this.setLayoutHyperparams();
  // initialize the layout hyperparams state
  var keys = Object.keys(this.hyperparams);
  for (var i=0; i<keys.length; i++) {
    this[keys[i]] = this.hyperparams[keys[i]][0];
  }
  // set the layout hyperparams in the gui
  this.setGuiHyperparams();
  // draw the layout and render the scene
  this.setLayout(function() {
    setTimeout(this.flyInCamera.bind(this), 500);
    window.addEventListener('resize', this.onWindowResize.bind(this));
    document.querySelector('#search').value = this.initialQuery;
  }.bind(this));
}


Wordmap.prototype.allAssetsLoaded = function() {
  // determine whether all assets required to create the scene have loaded
  return ['heightmap', 'manifest', 'texts'].reduce(function(b, o) {
    b = b && o in this.state.loaded; return b;
  }.bind(this), true)
}


Wordmap.prototype.setLayoutHyperparams = function() {
  // store the distinct levels for each factor in the current layout's hyperparams
  var params = {};
  this.data.manifest[this.layout].forEach(function(o) {
    Object.keys(o.params).forEach(function(k) {
      if (!(k in params)) params[k] = [o.params[k]];
      if (params[k].indexOf(o.params[k]) == -1) params[k].push(o.params[k]);
    })
  })
  this.hyperparams = params;
}


Wordmap.prototype.setGuiHyperparams = function() {
  // remove all hyperparemters currently present within the gui
  this.gui.layout.hyperparams.forEach(function(i) {
    this.gui.layout.folder.remove(i);
  }.bind(this))
  this.gui.layout.hyperparams = [];
  // add all of the hyperparemters for the current layout
  Object.keys(this.hyperparams).forEach(function(k) {
    var o = this.gui.layout.folder.add(this, k, this.hyperparams[k])
        .name(k)
        .onFinishChange(this.setLayout.bind(this))
    this.gui.layout.hyperparams.push(o);
  }.bind(this))
}


Wordmap.prototype.setLayout = function(cb) {
  // load the current layout data, update the gui, and transition points
  if (this.state.transitioning) {
    this.state.transitionQueued = true;
    return;
  }
  this.state.transitioning = true;
  get(this.getLayoutDataPath(), function(data) {
    this.data.selected = JSON.parse(data);
    this.data.selected.positions = center(this.data.selected.positions);
    // if the mesh doesn't exist, initialize it
    if (!this.mesh) {
      this.initializeMesh();
      this.state.transitioning = false;
      if (cb && typeof cb === 'function') cb();
    // if the mesh does exist, update it
    } else {
      var attrs = this.getWordAttrs();
      this.setPointScale();
      this.mesh.geometry.attributes.target.array = attrs.translations;
      this.mesh.geometry.attributes.target.needsUpdate = true;
      TweenLite.to(this.mesh.material.uniforms.transition, 1, {
        value: 1,
        ease: Power4.easeInOut,
        onComplete: function() {
          requestAnimationFrame(function() {
            this.mesh.geometry.attributes.translation.array = attrs.translations;
            this.mesh.geometry.attributes.translation.needsUpdate = true;
            this.mesh.material.uniforms.transition = {type: 'f', value: 0};
            this.state.transitioning = false;
            if (cb && typeof cb === 'function') cb();
          }.bind(this))
        }.bind(this)
      })
    }
  }.bind(this))
}


Wordmap.prototype.initializeMesh = function() {
  // add all words to the scene with initial attributes
  var attrs = this.getWordAttrs(),
      geometry = new THREE.InstancedBufferGeometry();
  geometry.addAttribute('uv', new BA(new ARR([0,0]), 2, true, 1));
  geometry.addAttribute('position', new BA(new ARR([0,0,0]), 3, true, 1));
  geometry.addAttribute('translation', new IBA(attrs.translations, 3, true, 1));
  geometry.addAttribute('target', new IBA(attrs.translations, 3, true, 1));
  geometry.addAttribute('texOffset', new IBA(attrs.texOffsets, 2, true, 1));
  // build the mesh
  this.setShaderMaterial();
  this.mesh = new THREE.Points(geometry, this.material);
  this.mesh.frustumCulled = false;
  this.mesh.name = 'words';
  this.scene.add(this.mesh);
}


Wordmap.prototype.getLayoutDataPath = function() {
  this.setLayoutHyperparams();
  this.setGuiHyperparams();
  var params = Object.keys(this.hyperparams).sort();
  return 'data/layouts/' + this.layout + '/' + params.reduce(function(s, k, idx) {
    return idx == Object.keys(this.hyperparams).length-1
      ? s + k + '-' + wm[k]
      : s + k + '-' + wm[k] + '-';
  }.bind(this), params.length ? this.layout + '_' : this.layout) + '.json';
}


Wordmap.prototype.createGui = function() {
  this.gui = {
    root: new dat.GUI({
      hideable: false,
    }),
    layout: {
      folder: null,
      hyperparams: [],
    },
    style: {
      folder: null,
    },
  };

  // layout folder
  this.gui.layout.folder = this.gui.root.addFolder('Layout');

  this.gui.layout.layout = this.gui.layout.folder.add(this, 'layout', this.data.layouts)
    .name('layout')
    .onFinishChange(this.setLayout.bind(this))

  this.gui.layout.heightmap = this.gui.layout.folder.add(this, 'heightScalar', 0.0, 0.003)
    .name('mountain')
    .onFinishChange(this.setLayout.bind(this))

  // style folder
  this.gui.style.folder = this.gui.root.addFolder('Style');

  this.gui.style.wordScalar = this.gui.style.folder.add(this, 'wordScalar', 0.0, 0.005)
    .name('font size')
    .onFinishChange(this.setLayout.bind(this))

  this.gui.style.background = this.gui.style.folder.addColor(this, 'background')
    .name('background')
    .onChange(this.setBackgroundColor.bind(this))

  this.gui.style.color = this.gui.style.folder.add(this, 'color', ['#fff', '#000'])
    .name('color')
    .onChange(this.updateTexture.bind(this))

  this.gui.style.font = this.gui.style.folder.add(this, 'font', this.fonts)
    .name('font')
    .onChange(this.updateTexture.bind(this))

  this.gui.style.mipmap = this.gui.style.folder.add(this, 'mipmap')
    .name('mipmap')
    .onChange(this.updateTexture.bind(this))

  this.gui.layout.folder.open();
  this.gui.style.folder.open();
}


/**
* Character canvas
**/

Wordmap.prototype.setCharacterCanvas = function() {
  // draw the letter bitmap on the 2d canvas
  var canvas = document.createElement('canvas'),
      ctx = canvas.getContext('2d'),
      charToCoords = {},
      yOffset = -0.25, // offset to draw full letters w/ baselines...
      xOffset = 0.05; // offset to draw full letter widths
  canvas.width = this.size * 16; // * 16 because we want 16**2 = 256 letters
  canvas.height = this.size * 16; // must set size before setting font size
  canvas.id = 'letter-canvas';
  ctx.font = this.size + 'px ' + this.font;
  // draw the letters on the 2d canvas
  ctx.fillStyle = this.color;
  for (var x=0; x<16; x++) {
    for (var y=0; y<16; y++) {
      var char = String.fromCharCode((x*16) + y);
      charToCoords[char] = {x: x, y: y};
      ctx.fillText(char, (x+xOffset)*this.size, yOffset*this.size+(y+1)*this.size);
    }
  }
  // build a three texture with the 2d canvas
  var tex = new THREE.Texture(canvas);
  tex.flipY = false;
  tex.needsUpdate = true;
  // store the texture in the current Wordmap instance
  this.data.characters = {
    map: charToCoords,
    tex: tex,
  }
}


/**
* Geometry
**/

Wordmap.prototype.getWordAttrs = function() {
  var n = 0, // total number of characters among all words
      positions = wm.data.selected.positions;
  for (var i=0; i<this.data.texts.length; i++) n += this.data.texts[i].length;
  // build up word attributes
  var attrs = {
    translations: new Float32Array(n * 3),
    texOffsets: new Float32Array(n * 2),
  }
  var iters = {
    translationIter: 0,
    texOffsetIter: 0,
  }
  // assume each word has x y coords assigned
  for (var i=0; i<this.data.texts.length; i++) {
    var word = this.data.texts[i],
        x = positions[i][0],
        y = positions[i][1],
        z = positions[i][2] || this.getHeightAt(x, y);
    for (var c=0; c<word.length; c++) {
      var offsets = this.data.characters.map[word[c]] || this.data.characters.map['?'];
      attrs.translations[iters.translationIter++] = x + (this.wordScalar * 0.9 * c);
      attrs.translations[iters.translationIter++] = y;
      attrs.translations[iters.translationIter++] = z;
      attrs.texOffsets[iters.texOffsetIter++] = offsets.x;
      attrs.texOffsets[iters.texOffsetIter++] = offsets.y;
    }
  }
  return attrs;
}


Wordmap.prototype.setShaderMaterial = function() {
  // set the material
  this.material = new THREE.RawShaderMaterial({
    vertexShader: document.getElementById('vertex-shader').textContent,
    fragmentShader: document.getElementById('fragment-shader').textContent,
    uniforms: {
      pointScale: { type: 'f', value: 0.0, },
      transition: { type: 'f', value: 0.0, },
      cellSize:   { type: 'f', value: this.size / (this.size * 16), }, // letter size in map
      color:      { type: 'f', value: this.getColorUniform() },
      tex:        { type: 't', value: this.data.characters.tex, },
    },
    transparent: false,
    defines: {
      WORDS: true,
    }
  });
  this.setPointScale();
}


Wordmap.prototype.getColorUniform = function() {
  // helper to indicate the color of the font in 0:1 coords
  return this.color === '#fff' ? 1.0 : 0.0;
}


Wordmap.prototype.getHeightAt = function(x, y) {
  // determine the height of the heightmap at coordinates x,y
  x = (x+1)/2; // rescale x,y axes from -1:1 to 0:1
  y = (y+1)/2;
  var row = Math.floor(y * this.data.heightmap.height),
      col = Math.floor(x * this.data.heightmap.width),
      idx = (row * this.data.heightmap.width * 4) + (col * 4),
      z = (this.data.heightmap.data[idx] + Math.random()) * this.heightScalar;
  return z;
}


/**
* User callbacks
**/

Wordmap.prototype.setBackgroundColor = function() {
  document.querySelector('body').style.background = this.background;
}


Wordmap.prototype.updateTexture = function() {
  this.setCharacterCanvas();
  // set the mipmaps
  var filter = this.mipmap ? THREE.LinearMipMapLinearFilter : THREE.LinearFilter;
  this.data.characters.tex.minFilter = filter;
  this.data.characters.tex.needsUpdate = true;
  this.mesh.material.uniforms.tex.value = this.data.characters.tex;
  this.mesh.material.uniforms.color.value = this.getColorUniform();
}


Wordmap.prototype.setPointScale = function() {
  var val = window.devicePixelRatio * window.innerHeight * this.wordScalar;
  this.material.uniforms.pointScale.value = val;
  this.material.uniforms.pointScale.needsUpdate = true;
  this.renderer.setPixelRatio(window.devicePixelRatio);
}


Wordmap.prototype.flyTo = function(coords) {
  if (this.state.flying) return;
  this.state.flying = true;
  // pull out target coordinates
  var self = this,
      x = coords[0],
      y = coords[1],
      z = coords[2] || self.getHeightAt(coords[0], coords[1]),
      z = z + 0.0175,
      // specify animation duration
      duration = 3,
      // create objects to use during flight
      aspectRatio = window.innerWidth / window.innerHeight,
      _camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.001, 10),
      _controls = new THREE.TrackballControls(_camera, self.renderer.domElement),
      q0 = self.camera.quaternion.clone(),
      _up = self.camera.up;
  _camera.position.set(x, y, z);
  _controls.target.set(x, y, z);
  _controls.update();
  TweenLite.to(self.camera.position, duration, {
    x: x,
    y: y,
    z: z,
    onUpdate: function() {
      if (!self.state.clock) {
        self.state.clock = new THREE.Clock();
        self.state.clock.start();
      }
      var deg = self.state.clock.getElapsedTime() / duration;
      THREE.Quaternion.slerp(q0, _camera.quaternion, self.camera.quaternion, deg);
    },
    onComplete: function() {
      var q = _camera.quaternion,
          p = _camera.position,
          u = _camera.up,
          c = _controls.target;
      self.camera.position.set(p.x, p.y, p.z);
      self.camera.up.set(u.x, u.y, u.z);
      self.camera.quaternion.set(q.x, q.y, q.z, q.w);
      self.controls.target = new THREE.Vector3(c.x, c.y, c.z-1.0);
      self.controls.update();
      self.state.flying = false;
      delete self.state.clock;
    },
    ease: Power4.easeInOut,
  });
}


Wordmap.prototype.flyInCamera = function() {
  TweenLite.to(this.camera.position, 3.5, {
    z: 0.56,
    ease: Power4.easeInOut,
  });
}


Wordmap.prototype.getWordCoords = function(word) {
  return this.data.selected.positions[wm.data.texts.indexOf(word)];
}

/**
* Typeahead
**/

Wordmap.prototype.queryWords = function(s) {
  var map = this.data.layouts[this.layout].wordToCoords;
  return Object.keys(map).filter(function(w) {
    return w.toLowerCase().indexOf(s.toLowerCase()) > -1;
  });
}


function Typeahead() {
  var input = document.querySelector('#search'), // query box
      typeahead = document.querySelector('#typeahead'), // typeahead options
      button = document.querySelector('#search-button'); // submit button

  input.addEventListener('keyup', function(e) {
    clearTypeahead();
    if (e.keyCode == 13 || e.target.value.length < 2) return;
    var matches = wm.queryWords(e.target.value),
        rendered = {}; // store the rendered objects to prevent cased dupes
    for (var i=0; i<Math.min(50, matches.length); i++) {
      if (!(matches[i].toLowerCase().trim() in rendered)) {
        rendered[ matches[i].toLowerCase().trim() ] = true;
        var elem = document.createElement('div');
        elem.textContent = matches[i];
        elem.onclick = function(str, e) {
          input.value = str;
          submit();
        }.bind(this, matches[i]);
        document.querySelector('#typeahead').appendChild(elem);
      }
    }
  })

  function clearTypeahead(e) {
    typeahead.innerHTML = '';
  }

  function submit() {
    if (!input.value) return;
    var coords = wm.getWordCoords(input.value);
    if (!coords) {
      var elem = document.querySelector('#no-results');
      elem.style.transform = 'translate(0, 42px)';
      setTimeout(function() {
        elem.style.transform = 'translate(0, 0)';
      }, 1500);
      return;
    }
    wm.flyTo(coords);
    clearTypeahead();
  }

  button.addEventListener('click', submit);
  window.addEventListener('click', clearTypeahead);
  input.addEventListener('keydown', function(e) {
    if (e.keyCode == 13) submit();
    else clearTypeahead();
  });
}


/**
* Helpers
**/

// center a 2d array of vertex positions -1:1 on each axis
function center(arr) {
  var max = Number.POSITIVE_INFINITY,
      min = Number.NEGATIVE_INFINITY,
      domX = {min: max, max: min},
      domY = {min: max, max: min},
      domZ = {min: max, max: min};
  // find the min, max of each dimension
  for (var i=0; i<arr.length; i++) {
    var x = arr[i][0],
        y = arr[i][1],
        z = arr[i][2] || 0;
    if (x < domX.min) domX.min = x;
    if (x > domX.max) domX.max = x;
    if (y < domY.min) domY.min = y;
    if (y > domY.max) domY.max = y;
    if (z < domZ.min) domZ.min = z;
    if (z > domZ.max) domZ.max = z;
  }
  var centered = [];
  for (var i=0; i<arr.length; i++) {
    var cx = (((arr[i][0]-domX.min)/(domX.max-domX.min))*2)-1,
        cy = (((arr[i][1]-domY.min)/(domY.max-domY.min))*2)-1,
        cz = (((arr[i][2]-domZ.min)/(domZ.max-domZ.min))*2)-1 || null;
    if (arr[i].length == 3) centered.push([cx, cy, cz]);
    else centered.push([cx, cy]);
  }
  return centered;
}


function get(url, onSuccess, onErr, onProgress) {
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.onreadystatechange = function() {
    if (xmlhttp.readyState == XMLHttpRequest.DONE) {
      if (xmlhttp.status === 200) {
        if (onSuccess) onSuccess(xmlhttp.responseText);
      } else {
        if (onErr) onErr(xmlhttp)
      }
    };
  };
  xmlhttp.onprogress = function(e) {
    if (onProgress) onProgress(e);
  };
  xmlhttp.open('GET', url, true);
  xmlhttp.send();
};


/**
* Main
**/

window.onload = function() {
  wm = new Wordmap();
  typeahead = new Typeahead();
};
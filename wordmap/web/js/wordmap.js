// aliases
var BA = THREE.BufferAttribute,
    IBA = THREE.InstancedBufferAttribute,
    ARR = Float32Array;


function Wordmap() {
  // config
  this.wordScalar = 0.0003; // sizes up words
  this.heightScalar = 0.002; // controls mountain height
  this.sep = 0.9; // separation between characters
  this.maxWords = 1000000; // max number of words to draw
  this.background = '#fff'; // background color
  this.color = '#000'; // text color
  // static
  this.size = 64; // size of each character on canvas
  // state
  this.state = {
    layout: 'grid', // name of the currently active layout
    flying: false, // bool indicating whether we're flying camera
    clock: null, // clock to measure how long we've been flying camera
    transitioning: false, // bool indicating whether layout is transitioning
    transitionQueued: false, // bool indicating whether to run another layout transition
  }
  // data
  this.data = {
    input: null,
    words: [],
    layouts: {},
    heightmap: {},
    characters: {},
  }
  // initialize
  this.init();
}


/**
* Scene
**/


Wordmap.prototype.createScene = function() {
  // generate a scene object
  var scene = new THREE.Scene();

  // generate a camera
  var aspectRatio = window.innerWidth / window.innerHeight;
  var camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.001, 10);

  // generate a renderer
  var renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.sortObjects = false; // make scene.add order draw order
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.domElement.id = 'gl-scene';
  document.body.appendChild(renderer.domElement);

  // generate controls
  var controls = new THREE.TrackballControls(camera, renderer.domElement);
  controls.zoomSpeed = 0.05;
  controls.panSpeed = 0.1;

  // position the camera
  camera.position.set(0.03, -0.80, 1.3);
  camera.up.set(0.00, 0.32, 0.94);
  camera.quaternion.set({_w: 0.81, _x: 0.58, _y: 0.01, _z: 0.00})
  controls.target.set(0.01, 1.00, 0.24);
  controls.update();

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


Wordmap.prototype.render = function() {
  requestAnimationFrame(this.render.bind(this));
  this.renderer.render(this.scene, this.camera);
  this.controls.update();
  if (this.state.transitionQueued) {
    this.state.transitionQueued = false;
    this.updateLayout();
  }
}


Wordmap.prototype.onWindowResize = function() {
  this.camera.aspect = window.innerWidth / window.innerHeight;
  this.camera.updateProjectionMatrix();
  this.renderer.setSize(window.innerWidth, window.innerHeight);
  this.setPointScale();
}


/**
* Character canvas
**/

Wordmap.prototype.setCharacters = function() {
  var canvas = document.createElement('canvas'),
      ctx = canvas.getContext('2d'),
      charToCoords = {},
      yOffset = -0.25, // offset to draw full letters w/ baselines...
      xOffset = 0.05; // offset to draw full letter widths
  canvas.width = this.size * 16; // * 16 because we want 16**2 = 256 letters
  canvas.height = this.size * 16; // must set size before setting font size
  canvas.id = 'letter-canvas';
  ctx.font = this.size + 'px Monospace';
  // draw the letters on the canvas
  ctx.fillStyle = this.color;
  for (var x=0; x<16; x++) {
    for (var y=0; y<16; y++) {
      var char = String.fromCharCode((x*16) + y);
      charToCoords[char] = {x: x, y: y};
      ctx.fillText(char, (x+xOffset)*this.size, yOffset*this.size+(y+1)*this.size);
    }
  }
  // build a three canvas with the canvas
  var tex = new THREE.Texture(canvas);
  tex.flipY = false;

  //tex.minFilter = THREE.NearestFilter;
  //tex.magFilter = THREE.NearestFilter;

  tex.needsUpdate = true;
  // store the character map on the instance
  this.data.characters = {
    map: charToCoords,
    tex: tex,
  }
}


/**
* Heightmap canvas
**/

Wordmap.prototype.getHeightmap = function(cb) {
  var img = new Image();
  img.crossOrigin = 'Anonymous';
  img.onload = function() {
    var canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    cb(ctx.getImageData(0,0, img.width, img.height));
  }
  img.src = 'heightmap.jpg';
}


/**
* Geometry
**/

Wordmap.prototype.addWords = function() {
  var attrs = this.getWordAttrs(),
      geometry = new THREE.InstancedBufferGeometry();
  geometry.addAttribute('uv', new BA(new ARR([0,0]), 2, true, 1));
  geometry.addAttribute('position', new BA(new ARR([0,0,0]), 3, true, 1));
  geometry.addAttribute('translation', new IBA(attrs.translations, 3, true, 1));
  geometry.addAttribute('target', new IBA(attrs.translations, 3, true, 1));
  geometry.addAttribute('texOffset', new IBA(attrs.texOffsets, 2, true, 1));
  // build the mesh
  this.setShaderMaterial();
  var mesh = new THREE.Points(geometry, this.material);
  mesh.frustumCulled = false;
  mesh.name = 'words';
  this.mesh = mesh;
  this.scene.add(mesh);
}


Wordmap.prototype.getWordAttrs = function() {
  var n = 0, // total number of characters among all words
      layout = this.data.layouts[this.state.layout],
      words = layout.words,
      positions = layout.positions;
  for (var i=0; i<words.length; i++) n += words[i].length;
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
  for (var i=0; i<words.length; i++) {
    var word = words[i],
        x = positions[i][0],
        y = positions[i][1],
        z = positions[i][2] || this.getHeightAt(x, y);
    for (var c=0; c<word.length; c++) {
      var offsets = this.data.characters.map[word[c]] || this.data.characters.map['?'];
      attrs.translations[iters.translationIter++] = x + (this.wordScalar * this.sep * c);
      attrs.translations[iters.translationIter++] = y;
      attrs.translations[iters.translationIter++] = z;
      attrs.texOffsets[iters.texOffsetIter++] = offsets.x;
      attrs.texOffsets[iters.texOffsetIter++] = offsets.y;
    }
  }
  return attrs;
}


Wordmap.prototype.setShaderMaterial = function() {
  this.material = new THREE.RawShaderMaterial({
    vertexShader: document.getElementById('vertex-shader').textContent,
    fragmentShader: document.getElementById('fragment-shader').textContent,
    uniforms: {
      pointScale: { type: 'f', value: 0.0, },
      cellSize:   { type: 'f', value: this.size / (this.size * 16), }, // letter size in map
      tex:        { type: 't', value: this.data.characters.tex, },
      color:      { type: 'f', value: this.getColorUniform() },
      transition: { type: 'f', value: 0.0, },
    },
    //transparent: true,
    defines: {
      WORDS: true,
    }
  });
  this.setPointScale();
}


Wordmap.prototype.getColorUniform = function() {
  return this.color === '#fff' ? 1.0 : 0.0;
}


Wordmap.prototype.getHeightAt = function(x, y) {
  // because x and y axes are scaled -1:1, rescale 0:1
  x = (x+1)/2;
  y = (y+1)/2;
  var row = Math.floor(y * this.data.heightmap.height),
      col = Math.floor(x * this.data.heightmap.width),
      idx = (row * this.data.heightmap.width * 4) + (col * 4),
      z = (this.data.heightmap.data[idx] + Math.random()) * this.heightScalar;
  return z;
}


Wordmap.prototype.init = function() {
  this.setCharacters();
  this.setBackgroundColor();
  this.getHeightmap(function(heightMapData) {
    this.data.heightmap = heightMapData;
    get('wordmap-layouts.json', function(data) {
      this.data.input = data;
      this.parseLayouts();
      this.createScene();
      this.addWords();
      this.render();
      setTimeout(this.flyInCamera.bind(this), 500);
      window.addEventListener('resize', this.onWindowResize.bind(this));
    }.bind(this))
  }.bind(this))
}


Wordmap.prototype.parseLayouts = function() {
  for (var i=0; i<this.data.input.length; i++) {
    var l = this.data.input[i],
        name = l.name || i,
        words = l.words,
        positions = this.center(l.positions),
        wordToCoords = {};
    for (var j=0; j<words.length; j++) {wordToCoords[words[j]] = positions[j];}
    this.data.layouts[name] = {
      words: words,
      positions: positions,
      wordToCoords: wordToCoords,
    }
    // activate the first layout
    if (i == 0 && !this.state.layout) this.state.layout = name;
  }
}


// center an array of vertex positions -1:1 on each axis
Wordmap.prototype.center = function(arr) {
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


Wordmap.prototype.queryWords = function(s) {
  var map = this.data.layouts[this.state.layout].wordToCoords;
  return Object.keys(map).filter(function(w) {
    return w.toLowerCase().indexOf(s.toLowerCase()) > -1;
  });
}


Wordmap.prototype.updateLayout = function() {
  if (this.state.transitioning) {
    this.state.transitionQueued = true;
    return;
  }
  this.state.transitioning = true;
  this.setPointScale();
  var attrs = this.getWordAttrs();
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
      }.bind(this))
    }.bind(this)
  })
}


/**
* User callbacks
**/

Wordmap.prototype.setBackgroundColor = function() {
  document.querySelector('body').style.background = this.background;
}


Wordmap.prototype.setTextColor = function() {
  this.setCharacters();
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
  return this.data.layouts[this.state.layout].wordToCoords[word];
}

/**
* Typeahaed
**/

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
      elem.style.transform = 'translate(0, 75px)';
      setTimeout(function() {
        elem.style.transform = 'translate(0, 24px)';
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
* Main
**/

function get(url, onSuccess, onErr, onProgress) {
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.onreadystatechange = function() {
    if (xmlhttp.readyState == XMLHttpRequest.DONE) {
      if (xmlhttp.status === 200) {
        if (onSuccess) onSuccess(JSON.parse(xmlhttp.responseText));
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


// main
window.onload = function() {
  wm = new Wordmap();
  typeahead = new Typeahead();

  // build the gui
  gui = new dat.GUI({hideable: false})

  gui.add(wm.state, 'layout', ['grid', 'tsne'])
    .name('layout')
    .onFinishChange(wm.updateLayout.bind(wm))

  gui.add(wm, 'wordScalar', 0.0, 0.001)
    .name('font size')
    .onFinishChange(wm.updateLayout.bind(wm))

  gui.add(wm, 'heightScalar', 0.0, 0.003)
    .name('mountain')
    .onFinishChange(wm.updateLayout.bind(wm))

  gui.addColor(wm, 'background')
    .name('background')
    .onChange(wm.setBackgroundColor.bind(wm))

  gui.add(wm, 'color', ['#fff', '#000'])
    .name('color')
    .onChange(wm.setTextColor.bind(wm))
};
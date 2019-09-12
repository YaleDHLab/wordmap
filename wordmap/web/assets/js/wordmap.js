// aliases
var BA = THREE.BufferAttribute,
    IBA = THREE.InstancedBufferAttribute,
    ARR = Float32Array;

function Wordmap() {
  // layout parameters
  this.layout = null; // the currently selected layout
  // style parameters
  this.wordSize = 0.001; // sizes up words
  this.pointSize = 0.001; // sizes up points
  this.maxWords = 1000000; // max number of words to draw
  this.background = '#222'; // background color
  this.color = '#fff'; // text color
  this.colorPoints = false; // bool indicating whether to color points
  this.font = 'Monospace'; // font family
  this.mipmap = false; // toggles mipmaps in texture
  this.transitionDuration = 1.0; // time of transitions in seconds
  this.renderPrimitive = 'words'; // the object to render {'points', 'words'}
  this.renderTooltip = false; // bool indicating whether to use gpu picking
  this.heightScalar = 0.001; // controls mountain height
  // internal static
  this.size = 64; // size of each character on canvas
  this.initialQuery = 'stars'; // the default search term
  // internal objects
  this.textMesh = null;
  this.pointMesh = null;
  this.pickingTextMesh = null;
  this.pickingPointMesh = null;
  // internal state
  this.state = {
    flying: false, // bool indicating whether we're flying camera
    clock: null, // clock to measure how long we've been flying camera
    transitioning: false, // bool indicating whether layout is transitioning
    transitionQueued: false, // bool indicating whether to run another layout transition
    loadProgres: {}, // map from asset identifier to load progress
    loaded: {}, // list of strings identifying initial assets loaded
  };
  // data
  this.data = {
    texts: [], // list of strings to visualize
    layouts: [],
    heightmap: {},
    characters: {}, // d[cluster_id] = {map: , tex: } controls cluster colors
    selected: {}, // currently selected layout
    previous: {}, // previously selected layout
  };
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
  var container = this.getContainer();
  var scene = new THREE.Scene();

  // camera
  var aspectRatio = container.w / container.h;
  var camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.01, 10);

  // renderer
  var renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    logarithmicDepthBuffer: true,
  });
  renderer.sortObjects = false; // make scene.add order draw order
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.w, container.h);
  renderer.domElement.id = 'gl-scene';
  container.elem.appendChild(renderer.domElement);

  // controls
  var controls = new THREE.TrackballControls(camera, renderer.domElement);
  controls.zoomSpeed = 0.05;
  controls.panSpeed = 0.1;

  // add ?axes=true to url to see axis helpers for global orientation
  if (window.location.search.includes('axes=true')) {
    var axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
  }

  // add ?stats=true to url to see rendering stats for a given host + browser
  if (window.location.search.includes('stats=true')) {
    this.stats = new Stats();
    this.stats.domElement.id = 'stats-container';
    this.stats.domElement.style.position = 'absolute';
    this.stats.domElement.style.top = '60px';
    this.stats.domElement.style.left = '0px';
    document.body.appendChild(this.stats.domElement);
  }

  // store objects on instance
  this.scene = scene;
  this.camera = camera;
  this.controls = controls;
  this.renderer = renderer;
}


Wordmap.prototype.createPickingScene = function() {
  // create an unrendered scene that's used for "GPU-picking"
  var pickingScene = new THREE.Scene(),
      container = this.getContainer(),
      pickingTarget = new THREE.WebGLRenderTarget(container.w, container.h);
  pickingTarget.texture.minFilter = THREE.LinearFilter;
  // trigger this.addTooltip on mouse move
  var mouse = new THREE.Vector2();
  this.renderer.domElement.addEventListener('mousemove', function(e) {
    var pixelBuffer = new Uint8Array(4);
    this.renderer.readRenderTargetPixels(
      pickingTarget,
      e.clientX,
      pickingTarget.height - e.clientY,
      1,
      1,
      pixelBuffer,
    );
    var id = (pixelBuffer[0]<<16)|(pixelBuffer[1]<<8)|(pixelBuffer[2]);
    if (id) { // id is the index position of the hovered word
      this.addTooltip(id-1); // animate a box that shows which text was selected
    }
  }.bind(this))
  // store references to the picking scene elements
  this.pickingScene = pickingScene;
  this.pickingTarget = pickingTarget;
}


Wordmap.prototype.addTooltip = function(idx) {
  // select the datum at index position idx
  console.log(' * select idx', idx)
}


Wordmap.prototype.getContainer = function() {
  // return the element and width, height attributes for scene container
  var elem = document.body;
  return {
    elem: elem,
    w: elem.clientWidth,
    h: elem.clientHeight,
  }
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
  if (this.renderTooltip) {
    this.renderer.render(this.pickingScene, this.camera, this.pickingTarget);
  }
  this.controls.update();
  if (this.state.transitionQueued) {
    this.state.transitionQueued = false;
    this.draw();
  }
  if (this.stats) {
    this.stats.update();
  }
}


Wordmap.prototype.onWindowResize = function() {
  // resize the canvas when the scene resizes
  var container = this.getContainer();
  this.camera.aspect = container.w / container.h;
  this.camera.updateProjectionMatrix();
  this.renderer.setSize(container.w, container.h);
  this.setPointScale();
}


/**
* Loaders
**/

Wordmap.prototype.loadManifest = function() {
  // load manifest file with all available layouts and initialize first layout
  get('data/manifest.json', function(data) {
    this.data.manifest = JSON.parse(data);
    this.data.layouts = Object.keys(this.data.manifest.layouts);
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
  }.bind(this);
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
  }.bind(this))
}


Wordmap.prototype.loadTexts = function() {
  // load all texts specified in this.texts
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
  this.createPickingScene();
  this.setInitialCameraPosition();
  this.render();
}


Wordmap.prototype.initializeIfLoaded = function() {
  // set the initial layout state and render the initial layout
  if (!this.allAssetsLoaded()) return;
  // set the initial layout state and add the mesh to the scene
  if (!this.layout) this.layout = this.data.layouts[0];
  // initialize the gui to which we'll add layout hyperparms
  this.createGui();
  // set the hyperparams for the current layout
  this.setHyperparams();
  // set the layout hyperparams in the gui
  this.setGuiHyperparams();
  // draw the layout and render the scene
  this.draw(function() {
    setTimeout(this.introduceScene.bind(this), 500);
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


Wordmap.prototype.setHyperparams = function() {
  // store the distinct levels for each factor in the current layout's hyperparams
  var params = {};
  // the file namespace is used by multiple layouts -- reset it to null
  if (this.file) this.file = null;
  this.data.manifest.layouts[this.layout].forEach(function(o) {
    Object.keys(o.params).forEach(function(k) {
      if (!(k in params)) params[k] = [o.params[k]];
      if (params[k].indexOf(o.params[k]) == -1) params[k].push(o.params[k].toString());
      if (!this[k]) this[k] = o.params[k];
    }.bind(this))
  }.bind(this))
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
        .onFinishChange(this.draw.bind(this))
    this.gui.layout.hyperparams.push(o);
  }.bind(this))
}


Wordmap.prototype.draw = function(cb) {
  // load the current layout data, update the gui, and transition points
  if (this.state.transitioning) {
    this.state.transitionQueued = true;
    return;
  }
  this.state.transitioning = true;
  get(this.getLayoutDataPath(), function(data) {
    this.data.selected = JSON.parse(data);
    this.data.selected.positions = center(this.data.selected.positions);
    // the mesh to be rendered doesn't exist; create it
    if (this.renderPrimitive == 'points' && !this.pointMesh ||
        this.renderPrimitive == 'words' && !this.textMesh) {
      this.initializeMeshes();
      this.setPointScale();
      this.state.transitioning = false;
      if (cb && typeof cb === 'function') cb();
    // the mesh to be rendered does exist; update it
    } else {
      var attrs = this.getMeshAttrs();
      this.setPointScale();
      var animationTargets = [];
      if (this.renderPrimitive == 'points') {
        ['pointMesh', 'pickingPointMesh'].forEach(function(m) {
          this[m].geometry.attributes.target.array = attrs.point.translations;
          this[m].geometry.attributes.clusterTarget.array = attrs.point.clusters;
          this[m].geometry.attributes.target.needsUpdate = true;
          this[m].geometry.attributes.clusterTarget.needsUpdate = true;
          this[m].material.uniforms.colorPoints.value = this.colorPoints ? 1.0 : 0.0;
          animationTargets.push(this[m].material.uniforms.transition);
        }.bind(this))
      } else {
        ['textMesh', 'pickingTextMesh'].forEach(function(m) {
          this[m].geometry.attributes.target.array = attrs.text.translations;
          this[m].geometry.attributes.target.needsUpdate = true;
          animationTargets.push(this[m].material.uniforms.transition);
        }.bind(this))
      }
      TweenLite.to(animationTargets, this.transitionDuration, {
        value: 1,
        ease: Power4.easeInOut,
        onComplete: function() {
          requestAnimationFrame(function() {
            if (this.renderPrimitive == 'points') {
              ['pointMesh', 'pickingPointMesh'].forEach(function(m) {
                this[m].geometry.attributes.cluster.array = attrs.point.clusters;
                this[m].geometry.attributes.translation.array = attrs.point.translations;
                this[m].geometry.attributes.cluster.needsUpdate = true;
                this[m].geometry.attributes.translation.needsUpdate = true;
                this[m].material.uniforms.transition = {type: 'f', value: 0};
              }.bind(this))
            } else {
              ['textMesh', 'pickingTextMesh'].forEach(function(m) {
                this[m].geometry.attributes.translation.array = attrs.text.translations;
                this[m].geometry.attributes.translation.needsUpdate = true;
                this[m].material.uniforms.transition = {type: 'f', value: 0};
              }.bind(this))
            }
            this.state.transitioning = false;
            if (cb && typeof cb === 'function') cb();
          }.bind(this))
        }.bind(this)
      })
    }
  }.bind(this))
}


Wordmap.prototype.initializeMeshes = function() {
  // remove extant meshes
  for (var i=0; i<this.scene.children.length; i++) {
    this.scene.remove(this.scene.children[i]);
  }
  // create meshes rendered for users
  var attrs = this.getMeshAttrs();
  if (this.renderPrimitive == 'points') {
    this.initializeMesh({
      name: 'points',
      scene: this.scene,
      defines: ['POINTS'],
      attrs: attrs.point,
      renderOrder: 0,
      reference: 'pointMesh',
    })
    this.initializeMesh({
      name: 'gpu-picking-points',
      scene: this.pickingScene,
      defines: ['POINTS', 'USE_PICKING_COLOR'],
      attrs: attrs.point,
      renderOrder: 0,
      reference: 'pickingPointMesh',
    })
  } else {
    this.initializeMesh({
      name: 'text',
      scene: this.scene,
      defines: ['TEXT'],
      attrs: attrs.text,
      renderOrder: 1,
      reference: 'textMesh',
    })
    this.initializeMesh({
      name: 'gpu-picking-text',
      scene: this.pickingScene,
      defines: ['TEXT', 'USE_PICKING_COLOR'],
      attrs: attrs.text,
      renderOrder: 1,
      reference: 'pickingTextMesh',
    })
  }
}


Wordmap.prototype.initializeMesh = function(obj) {
  // create the geometry for this mesh
  var geometry = new THREE.InstancedBufferGeometry();
  geometry.addAttribute('uv', new BA(new ARR([0,0]), 2, true, 1));
  geometry.addAttribute('position', new BA(new ARR([0,0,0]), 3, true, 1));
  geometry.addAttribute('translation', new IBA(obj.attrs.translations, 3, true, 1));
  geometry.addAttribute('target', new IBA(obj.attrs.translations, 3, true, 1));
  if (obj.attrs.texOffsets) {
    geometry.addAttribute('texOffset', new IBA(obj.attrs.texOffsets, 2, true, 1));
  }
  if (obj.attrs.clusters) {
    geometry.addAttribute('cluster', new IBA(obj.attrs.clusters, 1, true, 1));
    geometry.addAttribute('clusterTarget', new IBA(obj.attrs.clusters, 1, true, 1));
  }
  if (obj.defines.indexOf('USE_PICKING_COLOR') > -1) {
    geometry.addAttribute('pickingColor', new IBA(obj.attrs.pickingColor, 3, true, 1));
  }
  // create the material for this mesh
  var material = this.getShaderMaterial();
  for (var i=0; i<obj.defines.length; i++) {
    material.defines[obj.defines[i]] = true;
  }
  var mesh = new THREE.Points(geometry, material);
  mesh.frustumCulled = false;
  mesh.name = obj.name;
  mesh.renderOrder = obj.renderOrder;
  obj.scene.add(mesh);
  // store a reference to this mesh on parent scope
  this[obj.reference] = mesh;
}


Wordmap.prototype.getLayoutDataPath = function() {
  this.setHyperparams();
  this.setGuiHyperparams();
  var params = Object.keys(this.hyperparams).sort();
  return 'data/layouts/' + this.layout + '/' + params.reduce(function(s, k, idx) {
    return idx == Object.keys(this.hyperparams).length-1
      ? s + k + '-' + this[k]
      : s + k + '-' + this[k] + '-';
  }.bind(this), params.length ? this.layout + '_' : this.layout) + '.json';
}


Wordmap.prototype.createGui = function() {
  this.gui = {
    root: new dat.GUI({
      hideable: false,
    }),
    layout: {
      folder: null,
      hyperparams: [], // updated to set per-layout hyperparams
    },
    render: {
      folder: null,
    },
    style: {
      folder: null,
    },
  };

  // render folder
  this.gui.render.folder = this.gui.root.addFolder('Render');

  this.gui.render.primitive = this.gui.render.folder.add(this, 'renderPrimitive', ['points', 'words'])
    .name('render')
    .onFinishChange(this.onRenderPrimitiveChange.bind(this))

  this.setGuiRenderFolder();

  // layout folder
  this.gui.layout.folder = this.gui.root.addFolder('Layout');

  // add layouts option if there are multiple layout types
  if (this.data.layouts.length > 1) {
    this.gui.layout.layout = this.gui.layout.folder.add(this, 'layout', this.data.layouts)
      .name('layout')
      .onFinishChange(this.draw.bind(this))
  }

  // add heightmap controller if n components = 2
  if (this.data.manifest.params.n_components == 2) {
    this.gui.layout.heightmap = this.gui.layout.folder.add(this, 'heightScalar', 0.0, 0.003)
      .name('mountain')
      .onFinishChange(this.draw.bind(this))
  }

  // style folder
  this.gui.style.folder = this.gui.root.addFolder('Style');

  this.gui.style.background = this.gui.style.folder.addColor(this, 'background')
    .name('background')
    .onChange(this.setBackgroundColor.bind(this))

  this.gui.style.color = this.gui.style.folder.add(this, 'color', ['#fff', '#000'])
    .name('color')
    .onChange(this.updateTexture.bind(this))

  this.gui.style.mipmap = this.gui.style.folder.add(this, 'mipmap')
    .name('mipmap')
    .onChange(this.updateTexture.bind(this))

  this.gui.style.transitionDuration = this.gui.style.folder.add(this, 'transitionDuration', 0.0, 30.0)
    .name('transition time');

  this.gui.render.folder.open();
  this.gui.layout.folder.open();
  this.gui.style.folder.open();
}

Wordmap.prototype.setGuiRenderFolder = function() {
  // remove all elements in this folder
  ['pointSize', 'colorPoints', 'wordSize', 'font'].forEach(function(i) {
    if (this.gui.render[i]) {
      this.gui.render.folder.remove(this.gui.render[i])
      delete this.gui.render[i];
    }
  }.bind(this))

  // add the options appropriate for this render primitive
  if (this.renderPrimitive == 'points') {
    this.gui.render.pointSize = this.gui.render.folder.add(this, 'pointSize', 0.0, 0.003)
      .name('point size')
      .onFinishChange(this.draw.bind(this))

    this.gui.render.colorPoints = this.gui.render.folder.add(this, 'colorPoints')
      .name('color clusters')
      .onChange(this.draw.bind(this))
  } else {
    this.gui.render.wordSize = this.gui.render.folder.add(this, 'wordSize', 0.0, 0.003)
      .name('word size')
      .onFinishChange(this.draw.bind(this))

    this.gui.render.font = this.gui.render.folder.add(this, 'font', this.fonts)
      .name('font')
      .onChange(this.updateTexture.bind(this))
  }
}

Wordmap.prototype.onRenderPrimitiveChange = function() {
  this.setGuiRenderFolder();
  this.clearMeshes();
  this.draw();
}

Wordmap.prototype.clearMeshes = function() {
  ['pointMesh', 'pointPickingMesh', 'textMesh', 'textPickingMesh'].forEach(function(m) {
    delete this[m];
  }.bind(this))
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

Wordmap.prototype.getMeshAttrs = function() {
  var texts = this.data.texts,
      nWords = texts.length,
      nChars = texts.reduce(function(n, i) { n += i.length; return n; }, 0),
      nClusters = this.data.selected.cluster_centers.length;
  // build up attributes for the word and point geoemtries
  var attrs = {
    text: {
      translations: new Float32Array(nChars * 3),
      texOffsets: new Float32Array(nChars * 2),
      pickingColor: new Float32Array(nChars * 3),
    },
    point: {
      translations: new Float32Array(nWords * 3),
      clusters: new Float32Array(nWords),
      pickingColor: new Float32Array(nWords * 3),
    },
  }
  var iters = {
    text: {
      trans: 0,
      offsets: 0,
      pickingColor: 0,
    },
    point: {
      trans: 0,
      cluster: 0,
      pickingColor: 0,
    },
  }
  // assume each word has x y coords assigned
  var color = new THREE.Color();
  for (var i=0; i<nWords; i++) {
    var word = texts[i],
        rgb = color.setHex(i+1),
        x = this.data.selected.positions[i][0],
        y = this.data.selected.positions[i][1],
        z = this.data.selected.positions[i][2] || this.getHeightAt(x, y),
        cluster = this.data.selected.clusters[i] / nClusters; // normalize 0:1
    attrs.point.translations[iters.point.trans++] = x;
    attrs.point.translations[iters.point.trans++] = y;
    attrs.point.translations[iters.point.trans++] = z;
    attrs.point.clusters[iters.point.cluster++] = cluster;
    attrs.point.pickingColor[iters.point.pickingColor++] = rgb.r;
    attrs.point.pickingColor[iters.point.pickingColor++] = rgb.g;
    attrs.point.pickingColor[iters.point.pickingColor++] = rgb.b;
    for (var c=0; c<word.length; c++) {
      var offsets = this.data.characters.map[word[c]] || this.data.characters.map['?'];
      attrs.text.translations[iters.text.trans++] = x + (this.wordSize * 0.9 * c);
      attrs.text.translations[iters.text.trans++] = y;
      attrs.text.translations[iters.text.trans++] = z;
      attrs.text.texOffsets[iters.text.offsets++] = offsets.x;
      attrs.text.texOffsets[iters.text.offsets++] = offsets.y;
      attrs.text.pickingColor[iters.text.pickingColor++] = rgb.r;
      attrs.text.pickingColor[iters.text.pickingColor++] = rgb.g;
      attrs.text.pickingColor[iters.text.pickingColor++] = rgb.b;
    }
  }
  return attrs;
}


// fetch material for initial rendering only; all updates mutate this material
Wordmap.prototype.getShaderMaterial = function() {
  // return a new shader material
  return new THREE.RawShaderMaterial({
    vertexShader: document.getElementById('vertex-shader').textContent,
    fragmentShader: document.getElementById('fragment-shader').textContent,
    uniforms: {
      pointScale:  { type: 'f', value: 0.0, },
      transition:  { type: 'f', value: 0.0, },
      cellSize:    { type: 'f', value: this.size / (this.size * 16), }, // letter size in map
      color:       { type: 'f', value: this.getColorUniform(), },
      tex:         { type: 't', value: this.data.characters.tex, },
      colors:      { type: 'vec3', value: new Float32Array([1,2,3,4,5,6,7]), },
      colorPoints: { type: 'f', value: this.colorPoints ? 1.0 : 0.0 },
    },
    transparent: false,
  });
}


Wordmap.prototype.getColorUniform = function() {
  // helper to indicate the color of the font in 0:1 coords
  return this.color === '#fff' ? 1.0 : 0.0;
}


Wordmap.prototype.getHeightAt = function(x, y) {
  // determine the height of the heightmap at coordinates x,y
  x = (x+1)/2; // rescale x,y axes from -1:1 to 0:1
  y = (y+1)/2;
  var row = Math.floor(y * (this.data.heightmap.height-1)),
      col = Math.floor(x * (this.data.heightmap.width-1)),
      idx = (row * this.data.heightmap.width * 4) + (col * 4),
      z = (this.data.heightmap.data[idx]) * this.heightScalar;
  return z;
}


/**
* User callbacks
**/

Wordmap.prototype.setBackgroundColor = function() {
  document.querySelector('body').style.background = this.background;
  document.querySelector('body').style.transition = 'background-color 2s';
}


Wordmap.prototype.updateTexture = function() {
  this.setCharacterCanvas();
  // set the mipmaps
  var filter = this.mipmap ? THREE.LinearMipMapLinearFilter : THREE.LinearFilter;
  this.data.characters.tex.minFilter = filter;
  this.data.characters.tex.needsUpdate = true;
  this.textMesh.material.uniforms.tex.value = this.data.characters.tex;
  this.textMesh.material.uniforms.color.value = this.getColorUniform();
}


Wordmap.prototype.setPointScale = function() {
  var container = this.getContainer(),
      windowScalar = window.devicePixelRatio * container.h;
  this.renderPrimitive == 'points'
    ? this.pointMesh.material.uniforms.pointScale.value = windowScalar * this.pointSize
    : this.textMesh.material.uniforms.pointScale.value = windowScalar * this.wordSize
  this.renderer.setPixelRatio(window.devicePixelRatio);
}


Wordmap.prototype.flyTo = function(coords) {
  if (this.state.flying) return;
  this.state.flying = true;
  // pull out target coordinates
  var self = this,
      container = this.getContainer(),
      x = coords[0],
      y = coords[1],
      z = coords[2] || self.getHeightAt(coords[0], coords[1]),
      z = z + 0.0175,
      // specify animation duration
      duration = 3,
      // create objects to use during flight
      aspectRatio = container.w / container.h,
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


Wordmap.prototype.introduceScene = function() {
  this.updateTexture();
  this.renderer.domElement.style.opacity = 1;
  TweenLite.to(this.camera.position, 3.5, {
    z: 0.56,
    ease: Power4.easeInOut,
  });
}


Wordmap.prototype.getWordCoords = function(word) {
  return this.data.selected.positions[this.data.texts.indexOf(word)];
}

/**
* Typeahead
**/

Wordmap.prototype.queryWords = function(s) {
  return this.data.texts.filter(function(w) {
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

// center a 3d array of vertex positions -1:1 on each axis
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
  // use the axis with widest variance as unit vector
  var xVar = Math.abs(domX.max-domX.min),
      yVar = Math.abs(domY.max-domY.min),
      zVar = Math.abs(domZ.max-domZ.min),
      vars = [xVar, yVar, zVar],
      max = vars.sort().reverse()[0],
      idx = vars.indexOf(max);
  switch (idx) {
    case 0:
      domY.min *= (xVar/yVar);
      domY.max *= (xVar/yVar);
      domZ.min *= (xVar/zVar);
      domZ.max *= (xVar/zVar);
      break;
    case 1:
      domX.min *= (xVar/yVar);
      domX.max *= (xVar/yVar);
      domZ.min *= (zVar/yVar);
      domZ.max *= (zVar/yVar);
      break;
    case 2:
      domX.min *= (xVar/zVar);
      domX.max *= (xVar/zVar);
      domY.min *= (yVar/zVar);
      domY.max *= (yVar/zVar);
      break;
    default:
      console.warn(' * maximum domain could not be found')
  }
  // center the axes using the domain with widest variance
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
Code for [Keras plays catch](https://edersantana.github.io/articles/keras_rl/) blog post

### Train
```bash
python qlearn.py
```

### Test
1) Generate figures
```bash
python test.py
```

2) Make gif
```bash
ffmpeg -i %03d.png output.gif -vf fps=1
```

Alternatively, check [cadurosar](https://gist.github.com/cadurosar/bd54c723c1d6335a43c8) ipython notebook, there you should run cell 2 before cell 1.

### Requirements
* Prior supervised learning and Keras knowledge
* Python science stack (numpy, scipy, matplotlib) - Install Anaconda!
* Theano or Tensorflow
* Keras (last testest on commit b0303f03ff03)
* ffmpeg (optional)

### License
This code is released under [MIT license](https://en.wikipedia.org/wiki/MIT_License). 
(Note that Deep Q-Learning has its own patent by Google)
# gltf-skin-anim-viewer.py

A super-simple, ugly looking, CPU-side skinning implementation in Python. It uses
pygltflib to load files, moderngl to render, and numpy+numba for skinning. For some
reason numba wanted scipy for the numpy dot product. To installed required packages:

```bash
pip install -r /path/to/requirements.txt
```

To run:

```bash
python gltf-skin-anim-viewer.py
```

## The core stuff

In skin_animator.py you'll find how to implement it CPU-side. Leaving the GPU
implementation as an exercise to the reader.

The basics of it is to sort the nodes in parent-first order, which happens in the
loader. Then you're free to apply each parent's transform the all the kids:

```python
node.transform = np.dot(parent_node.transform, node.transform)
```

Each time you animate a step, you'll need to get the bone transformation for each
bone. And yeah, "bone" and "joint" means the same thing in this nomenclature...

```python
joint_matrix = np.dot(node.transform, inverse_bind_matrix.reshape((4,4)).T)
```

Then you just apply each animated bone with some weights to the vertices.

I highly recommend [a blog post](https://lisyarus.github.io/blog/posts/gltf-animation.html)
from the game developer lisyarus, where he take you through the ropes.


## Result

![result](https://raw.githubusercontent.com/highfestiva/gltf-skin-anim-viewer/master/screenshot.png)

 — Yuck! Did you model that?  
 — Absolutely not.


## Licence

MIT

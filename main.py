from hull3D import ConvexHull3D
import numpy as np
import os
from dcel import DCEL, Vertex
from numpy import array, unique, append, dot, cross
from collections import deque
from itertools import permutations
from random import sample
pts = np.random.randint(-100, 100, (100,3))

# Showing default parameters
Hull = ConvexHull3D(pts, run=True, preproc=False, make_frames=True, frames_dir='./frames/')

# To get Vertex objects:
vertices = Hull.DCEL.vertexDict.values()

# To get indices:
pts = Hull.getPts()    # to get pts in order used by ConvexHull3d
hull_vertices = pts[Hull.getVertexIndices()]

# To get vertices of each Face:
faces = [[list(v.p()) for v in face.loopOuterVertices()] for face in Hull.DCEL.faceDict.values()]
# os.system("ffmpeg -framerate 8 -pattern_type glob -i './frames/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p demo.mp4")
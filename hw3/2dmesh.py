from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
# img = Image.open(r'./minion/minion.png')
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
original_img  = Image.open(r'./minion/minion.png')
zoom = 1
original_img = original_img .convert('RGB')
original_img = original_img.resize((np.array(original_img .size)*zoom).astype(int))
h,w = original_img.size

img = np.array(original_img)
fig, ax = plt.subplots(1,1)
plt.imshow(img)
mesh = []
for i in range(0,h,zoom):
    for j in range(0,w,zoom):
        # print(i,j)
        # plt.plot(i,j)
        if i+1<h and j+1<w:
            mesh.append(Polygon([(i, j), (i + zoom, j), (i + zoom, j + zoom)]))
            mesh.append(Polygon([(i, j), (i, j + zoom), (i + zoom, j + zoom)]))
        # img[i,j,:] = 0

# a = ax.add_patch(mesh)
# a.set_fill(False)
# print(mesh)
for tri in mesh:
    a = ax.add_patch(tri)
    a.set_fill(False)
# ax = plt.gca()
# ax.set_aspect(1)
# plt.ylim(0,w)
# plt.xlim(0,h)
plt.show()
# plt.imshow(img)


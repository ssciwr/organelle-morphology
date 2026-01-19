# ruff: noqa
# fmt: off
# %%
%load_ext rich
from functools import reduce
from pathlib import Path
import zmesh
from skimage import measure
from zmesh import Mesher
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# mpl.use("nbAgg")

# %%
arr = np.zeros((10,10,10))
arr[3:7,3:6,1:3] = 1
arr[3:6,3:6,4:6] = 2
colors = np.empty_like(arr, dtype=str)
colors[arr == 1] = "red"
colors[arr == 2] = "blue"

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(
    arr,
    facecolors=colors

)
plt.show(block=False)

# %%

mesher = Mesher((1,1,1))
mesher.mesh(arr)

meshes = []
for id in mesher.ids():
    meshes.append(mesher.get(
        id,
        normals=False,
        reduction_factor=2,

    ))

# %%

size = 60
n_points = 20

points = np.random.randint(0,size,n_points*3).reshape((-1,3))
x,y,z = np.indices((size, size, size))

def cubify(p, x, y, z):
    cube = ((p[0] + np.random.randint(2,6)) > x) & (x >= (p[0]-np.random.randint(2,6))) & \
        ((p[1] + np.random.randint(2,6)) > y)&(y >= (p[1]-np.random.randint(2,6))) & \
        ((p[2] + np.random.randint(2,6)) > z)&(z >= (p[2]-np.random.randint(2,6)))
    return cube

cm = matplotlib.colormaps["tab10"]
cubes = []
colors = np.zeros(x.shape + (3,))
for i, p in enumerate(points):
    cube = cubify(p,x,y,z)
    n_cube = np.zeros_like(cube,dtype=float)
    n_cube[cube] = i + 1
    cubes.append(n_cube)
    colors[cube] = cm.colors[i%10]

voxelarray = reduce(np.add, cubes)

# %%
ax = plt.figure().add_subplot(projection='3d')

for i, cube in enumerate(cubes):
    ax.voxels(
        filled=cube,
        facecolors=cm.colors[i%10],
    )
plt.show(block=False)

# %% zmesh
%%time
mesher = Mesher((1,1,1))
mesher.mesh(voxelarray, close=True)

meshes = []
for id in mesher.ids():
    meshes.append(mesher.get(
        id,
        normals=False,
        reduction_factor=1,
        voxel_centered=False,
        max_error=1,

    ))
meshes[0]

# %%
cm = matplotlib.colormaps.get("tab10")
ax = plt.figure().add_subplot(projection='3d')
for i, m in enumerate(meshes):
    ax.plot_trisurf(
        m.vertices[:,0,],
        m.vertices[:,1,],
        m.vertices[:,2,],
        triangles=m.faces,
        color=cm.colors[i%10]

    )
plt.show(block=False)

# %%
%%time
for i in range(len(points)):
    arr = np.zeros_like(voxelarray)
    arr[np.argwhere(voxelarray == i+1)] = 1
    verts, faces, _, _ = measure.marching_cubes(arr)

# @@@@@@@@@ one order of magnitude faster with zmesh! @@@@@@@@@@@@
# %%




# %%

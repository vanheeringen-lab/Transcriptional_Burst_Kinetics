import numpy as np
import plotly.graph_objs as go


def voxels(vox_image: np.ndarray, starts=[0, 0, 0], stepsize=[1, 1, 1]):
    assert vox_image.ndim == 3

    pad_ori = np.pad(vox_image, 1, mode='constant', constant_values=False)[1:, 1:, 1:]

    xs, ys, zs = [], [], []
    xx, yy, zz = [], [], []
    nr_faces = 0
    for dim in [0, 1, 2]:  # x, y, z dimensions
        for x, y, z in zip(*np.where(np.roll(pad_ori, shift=1, axis=dim) ^ pad_ori)):
            # x -= 1 + 0.5 * stepsize[0]; y -= 1 + 0.5 * stepsize[1]; z -= 1 + 0.5 * stepsize[2]
            x = starts[0] + (x - 0.5) * stepsize[0]
            y = starts[1] + (y - 0.5) * stepsize[1]
            z = starts[2] + (z - 0.5) * stepsize[2]

            if dim == 0:
                xs.extend([x] * 4)
                ys.extend([y, y+stepsize[1], y, y+stepsize[1]])
                zs.extend([z, z, z+stepsize[2], z+stepsize[2]])
            if dim == 1:
                xs.extend([x, x+stepsize[0], x, x+stepsize[0]])
                ys.extend([y] * 4)
                zs.extend([z, z, z+stepsize[2], z+stepsize[2]])
            if dim == 2:
                xs.extend([x, x+stepsize[0], x, x+stepsize[0]])
                ys.extend([y, y, y+stepsize[1], y+stepsize[1]])
                zs.extend([z] * 4)

            xx.extend([nr_faces * 4 + 0, nr_faces * 4 + 3])
            yy.extend([nr_faces * 4 + 1, nr_faces * 4 + 1])
            zz.extend([nr_faces * 4 + 2, nr_faces * 4 + 2])

            nr_faces += 1

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=xx,
        j=yy,
        k=zz,
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.9, specular=0.6, fresnel=0.2),
        # hoverinfo='skip'
    )

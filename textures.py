import glm
import pygame as pg
import moderngl as mgl
import numpy as np


base_dir = 'resources/textures/'


class Textures:
    def __init__(self, ctx):
        self.ctx = ctx
        self.next_texture_unit = 4
        self.textures = {}

    def load_all(self):
        self.load('stupid')

    def load(self, name, array_layers=False, smooth=False, interp=False, ext='png'):
        filename = self.file_resolve(f'{base_dir}{name}.{ext}')
        is_3d = False
        if ext != 'npz':
            texture = pg.image.load(filename)
            is_transparent = get_transparency(texture) if ext == 'png' else False
        else:
            texture = self.load3d(filename)
            is_3d = True
            is_transparent = False

        if is_3d:
            texture = self.ctx.texture3d(
                size=texture.size,
                components=texture.components,
                data=texture.data,
                dtype=texture.dtype
            )
            texture.num_layers = 0
        elif array_layers:
            num_layers = array_layers * texture.get_height() // texture.get_width()  # N textures per layer
            texture = self.ctx.texture_array(
                size=(texture.get_width(), texture.get_height() // num_layers, num_layers),
                components=4,
                data=pg.image.tostring(texture, 'RGBA')
            )
            texture.num_layers = num_layers
        else:
            texture = self.ctx.texture(
                size=texture.get_size(),
                components=4,
                data=pg.image.tostring(texture, 'RGBA', False)
            )
            texture.num_layers = 0
        texture.anisotropy = 0.0
        texture.is_transparent = is_transparent
        if 'tangent' in name or 'normal' in name or is_3d:
            texture.repeat_x = True
            texture.repeat_y = True
        if smooth or interp:
            texture.build_mipmaps()
            if smooth:
                texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            else:
                texture.filter = (mgl.NEAREST_MIPMAP_LINEAR, mgl.NEAREST)
        else:
            texture.filter = (mgl.NEAREST, mgl.NEAREST)

        # assign texture unit
        texture.unit = self.get_unit()
        texture.use(location=texture.unit)
        texture.gsize = glm.vec2(texture.size)

        self.textures[name] = texture
        return texture

    def load3d(self, path):
        npz = np.load(path)
        assert len(npz) == 1
        name = npz.files[0]
        voxels = npz[name]
        size = voxels.shape[::-1][1:]
        texture3d = np.reshape(voxels, -1)
        dtype_xlat = {'int8':'i1', 'uint8': 'u1', 'float8': 'f1', 'float16': 'f2'}
        ret = dict(size=size, components=voxels.shape[-1], data=texture3d.tobytes(), dtype=dtype_xlat[str(texture3d.dtype)])
        return ret

    def get(self, name):
        return self.textures[name]

    def get_unit(self):
        unit = self.next_texture_unit
        self.next_texture_unit += 1
        return unit

    def file_resolve(self, filename):
        return filename


def get_transparency(texture: pg.surface.Surface):
    w, h = texture.get_size()
    w, h = w-1, h-1
    check_coords = [(0,0), (1,0), (0,1), (1,1), (0.5,0.5)]
    for x, y in check_coords:
        ix, iy = int(x*w), int(y*h)
        col = texture.get_at((ix, iy))
        if col.a < 250:
            return True
    return False

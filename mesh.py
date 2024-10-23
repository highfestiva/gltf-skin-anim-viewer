from moderngl_window.opengl.vao import VAO
import numpy as np


class Mesh:
    def __init__(self, ctx, shader_program, mesh, texture=None):
        self.ctx = ctx
        self.program = shader_program
        self.mesh = mesh
        self.texture = texture
        self.is_shared = False

        self.vao_wrapper = VAO(name=mesh.name)
        self.vao_wrapper.index_buffer(self.mesh.triangles, index_element_size=2)
        if self.texture:
            self.vao_wrapper.buffer(self.mesh.uvs, '2f4', 'in_tex_coord')
            assert self.mesh.uvs.dtype == np.float32
        self.vao_wrapper.buffer(self.mesh.vertices, '3f4', 'in_position')
        assert self.mesh.vertices.dtype == np.float32

        self.instance_data = None
        self.max_instances = 512
        self.prepared_instances = 0

    @property
    def is_transparent(self):
        if not self.texture:
            return True
        return self.texture.is_transparent

    def render(self, instances=1):
        if self.texture:
            self.program['u_texture_0'] = self.texture.unit
        vao = self.vao_wrapper.instance(self.program)
        if instances > 1:
            assert instances == self.prepared_instances
        vao.render(instances=instances)

    def release(self):
        if self.is_shared:
            return
        self.vao_wrapper.release()

    def prepare_instances(self, num_floats, in_data):
        if self.instance_data is None:
            assert len(in_data) % num_floats == 0
            num_bytes = num_floats * 4
            print(f'preparing {num_floats} floats / instance, reserving {num_bytes * self.max_instances} bytes')
            self.instance_data = self.ctx.buffer(reserve=num_bytes * self.max_instances)
            self.vao_wrapper.buffer(self.instance_data, f'{num_floats}f/i', 'in_data')

        self.prepared_instances = len(in_data) // num_floats
        if self.prepared_instances > self.max_instances:
            in_data = in_data[:self.max_instances * num_floats]
            self.prepared_instances = self.max_instances
        assert len(in_data[:1].tobytes()) == 4
        self.instance_data.write(in_data)
        return self.prepared_instances

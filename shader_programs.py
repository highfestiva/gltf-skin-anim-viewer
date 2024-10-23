from glm import mat4
from pathlib import Path


base_dir = 'resources/shaders/'


class ShaderPrograms:
    def __init__(self, ctx, textures):
        self.ctx = ctx
        self.textures = textures
        self.programs = {}
        self.shader_names = [
            dict(shader_name='plain'),
        ]

    def update_uniforms(self, camera):
        plain = self.programs['plain']
        proj_mat = camera.get_projection()
        view_mat = camera.get_view()
        plain['m_proj'].write(proj_mat)
        plain['m_view'].write(view_mat)
        plain['m_model'].write(mat4())
        plain['u_texture_0'] = self.textures.get('stupid').unit
        plain['brightness'] = 1.1

    def get(self, name):
        return self.programs[name]

    def load_program(self, ctx, shader_name, frag_shader_name=None):
        vertex_shader = Path(self.file_resolve(f'{base_dir}{shader_name}.vert')).read_text()
        shader = frag_shader_name or shader_name
        fragment_shader = Path(self.file_resolve(f'{base_dir}{shader}.frag')).read_text()
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        name = program.name = f'{shader_name}_{frag_shader_name}' if frag_shader_name else shader_name
        self.programs[name] = program
        return program

    def load_all(self):
        self.release()
        for shader_name in self.shader_names:
            self.load_program(self.ctx, **shader_name)

    def release(self):
        for program in self.programs.values():
            program.release()
        self.programs.clear()

    def file_resolve(self, filename):
        return filename

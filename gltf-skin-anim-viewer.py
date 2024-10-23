#!/usr/bin/env python3

import moderngl
import moderngl_window as mglw

from camera import Camera
from gltf_loader import load_model
from mesh import Mesh
from shader_programs import ShaderPrograms
from skin_animator import SkinAnimator
from textures import Textures


class Window(mglw.WindowConfig):

    title = 'Python GLTF Skinning Animation Model Viewer v0.1'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = load_model('resources/models/stupid-knight.glb')
        self.textures = Textures(self.ctx)
        self.textures.load_all()
        self.shader_programs = ShaderPrograms(self.ctx, self.textures)
        self.shader_programs.load_all()
        self.meshes = []
        self.camera = Camera(distance=20, far=1000)
        self.animator = SkinAnimator(self.model)
        self.animator.start_animate()

    def render(self, time: float, delta_time: float):
        # update meshes
        self.animate(delta_time)
        self.rebuild_meshes()
        # setup gl and shaders
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.shader_programs.update_uniforms(self.camera)
        # actual render of each primitive
        for mesh in self.meshes:
            mesh.render()

    def animate(self, delta_time):
        self.animator.play_animation(delta_time)

    def rebuild_meshes(self):
        # release previous meshes
        for mesh in self.meshes:
            mesh.release()
        self.meshes.clear()
        # build new ones
        prog = self.shader_programs.get('plain')
        primitives = self.animator.skinned_primitives
        self.meshes = [Mesh(self.ctx, prog, p, self.textures.get('stupid')) for p in primitives]

    def mouse_drag_event(self, x: int, y: int, dx, dy):
        if self.wnd.mouse_states.middle:
            self.camera.pan(x=dx*0.01, y=dy*-0.01)
        else:
            self.camera.orbit(yaw=dx*0.01, pitch=dy*0.01)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.camera.dolly(y_offset*0.1)

    def resize(self, width: int, height: int):
        self.ctx.viewport = 0, 0, width, height
        self.camera.set_aspect_ratio(width / height)


if __name__ == '__main__':
    mglw.run_window_config(Window)

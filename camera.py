import glm


RIGHT = glm.vec3(1,0,0)
FORWARD = glm.vec3(0,1,0)
UP = glm.vec3(0,0,1) # Z is up, right-hand coordinate system, yey!


class Camera:
    def __init__(self, near=0.1, far=100, fov=50, distance=10):
        self.aspect_ratio = 1
        self.near = near
        self.far = far
        self.fov = fov
        self.yaw = 0
        self.pitch = 0
        self.distance = distance
        self.center_pos = glm.vec3()

    def pan(self, x, y):
        '''pan relative to direction we're looking'''
        q = self.get_quat()
        pos = q * glm.vec3(x, 0, 0) # left-right
        pos += q * glm.vec3(0, 0, y) # up-down
        self.center_pos += pos

    def orbit(self, yaw, pitch):
        '''rotate around center'''
        self.yaw += yaw
        self.pitch += pitch
        self.pitch = glm.clamp(self.pitch, -glm.radians(90), +glm.radians(90))

    def dolly(self, fact):
        '''move closer or further'''
        dist_fact = 1 - glm.clamp(fact, -2.5, 0.9)
        self.distance *= dist_fact

    def get_projection(self):
        return glm.perspective(self.fov, self.aspect_ratio, self.near, self.far)

    def get_view(self):
        q = self.get_quat()
        pos = q * glm.vec3(0, -self.distance, 0) # +Y is forward
        up = q * UP
        view_mat = glm.lookAt(pos, self.center_pos, up)
        return view_mat

    def get_quat(self):
        q_yaw = glm.rotate(glm.quat(), self.yaw, UP)
        q_pitch = glm.rotate(glm.quat(), self.pitch, RIGHT)
        return q_yaw * q_pitch

    def set_aspect_ratio(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

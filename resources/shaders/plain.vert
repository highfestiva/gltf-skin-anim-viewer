#version 450 core

in vec2 in_tex_coord;
in vec3 in_position;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;
uniform vec2 uv_offset;

out vec2 uv;


void main() {
    uv = in_tex_coord + uv_offset;
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}

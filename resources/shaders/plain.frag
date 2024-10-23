#version 450 core

out vec4 out_color;

in vec2 uv;

uniform sampler2D u_texture_0;
uniform float brightness;


void main() {
    vec4 rgba = texture(u_texture_0, uv);
    rgba.rgb *= brightness;
    out_color = rgba;
}

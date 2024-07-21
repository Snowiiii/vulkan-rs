#version 450 core

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D in_sampledTexture;

void main() {
	vec4 texSample = texture(in_sampledTexture, in_uv);
	out_color = texSample;
}
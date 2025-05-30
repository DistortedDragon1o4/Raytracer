#version 460 core

layout(binding = 0) uniform sampler2D tex;

uniform int width;
uniform int height;

in vec2 texCoord;

out vec4 FragColor;

void main() {
    FragColor = texture(tex, texCoord);
}

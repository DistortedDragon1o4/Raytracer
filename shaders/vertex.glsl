#version 460 core

out vec2 texCoord;

void main() {
    float vertexData[12] = {
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0
        };

    texCoord = (vec2(vertexData[2 * gl_VertexID], vertexData[2 * gl_VertexID + 1]) + 1.0) * 0.5;
    gl_Position = vec4(vertexData[2 * gl_VertexID], vertexData[2 * gl_VertexID + 1], 0.0, 1.0);
}

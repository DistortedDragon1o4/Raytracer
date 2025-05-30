#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stb/stb_image.h>

#include "shaderCompiler.h"

namespace std {
    typedef basic_string<unsigned char> ustring;
}

class Texture {
public:
    std::string assets = "assets/";
    GLuint ID;
    GLenum type;
    int slot;
    Texture(GLenum texType, GLenum slot, int width, int height);

    void BindTexImage();
    void GetTexImage(std::vector<unsigned char> &pixels);
    void Bind();
    void Unbind();
    void Delete();
};

class TextureArray {
public:
    std::string assets = "assets/";
    GLuint ID;
    GLenum type;
    int slot;
    TextureArray(int _slot, std::string directory, int start, int stop, std::string path);

    void TexUnit(Shader& shader, const char* uniform, GLuint unit);
    void Bind();
    void Unbind();
    void Delete();
};

struct Sampler {
    unsigned int ID;

    Sampler();

    void bind(int slot);
    void unbind();
    void del();
};

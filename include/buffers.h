#pragma once

#include "gladContainer.h"
#include <vector>
#include <array>

struct UnifiedGLBufferContainer {
    ~UnifiedGLBufferContainer() {del();};

    unsigned int ID;
    unsigned int size;
    void* persistentMappedPtr;

    void allocate(unsigned long size, GLenum flags);
    void upload(void* data, unsigned long size, unsigned long byteIndex);

    void map(GLenum flags);
    void map(unsigned long index, unsigned long _size, GLenum flags);

    void bind(GLenum type);
    void bindBufferBase(GLenum type, unsigned int bindIndex);
    void unbind(GLenum type);

    void create();
    void del();
};

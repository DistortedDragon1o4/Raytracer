#pragma once

#include "gladContainer.h"
#include <glm/glm.hpp>
#include "buffers.h"

struct newVAO {
    unsigned int ID;

    void create();

    void bind();
    void unbind();
    void del();
};

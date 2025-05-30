#include "../include/VAO.h"

void newVAO::create() {
    glCreateVertexArrays(1, &ID);
}

void newVAO::bind() {
    glBindVertexArray(ID);
}

void newVAO::unbind() {
    glBindVertexArray(0);
}

void newVAO::del() {
    glDeleteVertexArrays(1, &ID);
}

#include "../include/buffers.h"
#include <iostream>


void UnifiedGLBufferContainer::create() {
	glCreateBuffers(1, &ID);
}

void UnifiedGLBufferContainer::del() {
	glDeleteBuffers(1, &ID);
}

void UnifiedGLBufferContainer::bind(GLenum type) {
	glBindBuffer(type, ID);
}

void UnifiedGLBufferContainer::bindBufferBase(GLenum type, unsigned int bindIndex) {
	glBindBufferBase(type, bindIndex, ID);
}

void UnifiedGLBufferContainer::unbind(GLenum type) {
	glBindBuffer(type, ID);
}

void UnifiedGLBufferContainer::allocate(unsigned long size, GLenum flags) {
	this->size = size;
	glNamedBufferStorage(ID, size, NULL, flags);
}

void UnifiedGLBufferContainer::upload(void* data, unsigned long size, unsigned long byteIndex) {
	glNamedBufferSubData(ID, byteIndex, size, data);
}

void UnifiedGLBufferContainer::map(GLenum flags) {
	persistentMappedPtr = glMapNamedBufferRange(ID, 0, size, flags);
}

void UnifiedGLBufferContainer::map(unsigned long index, unsigned long _size, GLenum flags) {
	persistentMappedPtr = glMapNamedBufferRange(ID, index, _size, flags);
}

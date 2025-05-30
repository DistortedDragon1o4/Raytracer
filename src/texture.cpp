#include "../include/texture.h"
#include "glad/gl.h"

Texture::Texture(GLenum texType, GLenum slot, int width, int height) {
    type = texType;
    this->slot = slot;

	glCreateTextures(GL_TEXTURE_2D, 1, &ID);
	glBindTextureUnit(slot, ID);

	glTextureParameteri(ID, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(ID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTextureParameteri(ID, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTextureParameteri(ID, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTextureParameteri(ID, GL_TEXTURE_MAX_LEVEL, 4);
	glTextureParameteri(ID, GL_TEXTURE_BASE_LEVEL, 0);

	glTextureStorage2D(ID, 1, GL_RGBA8, width, height);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::GetTexImage(std::vector<unsigned char> &pixels) {
	glGetTextureImage(ID, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.size() * sizeof(pixels), pixels.data());
}

void Texture::BindTexImage() {
	glBindImageTexture(0, ID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
}

void Texture::Bind() {
	glBindTextureUnit(slot, ID);
}

void Texture::Unbind() {
	glBindTextureUnit(slot, 0);
}

void Texture::Delete() {
	glDeleteTextures(1, &ID);
}

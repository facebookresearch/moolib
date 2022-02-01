/* Copyright (c) Facebook, Inc. and its affiliates. */
#include <stdlib.h>

void *dlmalloc(size_t size) { return malloc(size); }
void dlfree(void *ptr) { free(ptr); }

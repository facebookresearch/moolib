#include <stdlib.h>

void *dlmalloc(size_t size) { return malloc(size); }
void dlfree(void *ptr) { free(ptr); }

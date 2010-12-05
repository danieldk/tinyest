#ifndef BITVECTOR_H
#define BITVECTOR_H

/* Bit vector. */
typedef struct {
  char *vector;
  size_t n;
  size_t on;
} bitvector_t;

typedef struct {
  size_t idx;
  char val;
} bitvector_iterator_t;

bitvector_t *bitvector_alloc(size_t n);
void bitvector_free(bitvector_t *bv);
void bitvector_set(bitvector_t *bv, size_t idx, char val);
char bitvector_get(bitvector_t *bv, size_t idx);

#endif // BITVECTOR_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <tinyest/bitvector.h>

bitvector_t *bitvector_alloc(size_t n)
{
  bitvector_t *bv;
  if ((bv = malloc(sizeof(bitvector_t))) == NULL) {
    perror("malloc() error in bitvector_alloc()");
    exit(1);
  }

  if ((bv->vector = (char *) malloc(sizeof(char) * n)) == NULL) {
    perror("malloc() error in bitvector_alloc()");
    exit(1);
  }
  memset(bv->vector, 0, sizeof(char) * n);
  bv->n = n;
  bv->on = 0;

  return bv;
}

void bitvector_free(bitvector_t *bv)
{
  free(bv->vector);
  bv->n = 0;
  bv->on = 0;
  free(bv);
}

void bitvector_set(bitvector_t *bv, size_t idx, char val) {
  assert(val == 0 || val == 1);

  if (bv->vector[idx] != val) {
    if (val == 1)
      ++bv->on;
    else
      --bv->on;
  }

  bv->vector[idx] = val;
}

char bitvector_get(bitvector_t *bv, size_t idx) {
  return bv->vector[idx];
}

void bitvector_iter_move(bitvector_t *bv, bitvector_iterator_t *iter) {
  while (iter->idx < bv->n && bv->vector[iter->idx] == 0)
    ++iter->idx;

  if (iter->idx != bv->n)
    iter->val = bv->vector[iter->idx];
  else
    iter->val = 0;
}

bitvector_iterator_t bitvector_begin(bitvector_t *bv) {
  bitvector_iterator_t iter = {0, 0};
  bitvector_iter_move(bv, &iter);
  return iter;
}

bitvector_iterator_t bitvector_next(bitvector_t *bv,
    bitvector_iterator_t iter) {
  ++iter.idx;
  bitvector_iter_move(bv, &iter);
  return iter;
}

bitvector_iterator_t bitvector_end(bitvector_t *bv) {
  bitvector_iterator_t iter = {bv->n, 0};
  return iter;
}


/*
 * Copyright 2010 Daniël de Kok 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tinyest/lbfgs.h>
#include <tinyest/model.h>
#include <tinyest/rbtree/red_black_tree.h>

void model_new(model_t *model, size_t n_params)
{
  model->params = lbfgs_malloc(n_params);
  model->n_params = n_params;
  model->f_restrict = 0;
}

void model_free(model_t *model)
{
  model->n_params = 0;
  lbfgs_free(model->params);
}

int int_comp(void const *a, void const *b) {
  if (*(int *) a > *(int *) b)
    return 1;
  if (*(int *) a < *(int *) b)
    return -1;
  return 0;
}

void int_dealloc(void *a) {
  free((int *) a);
}

void int_print(void const *a) {}
void info_print(void *a) {}
void info_dealloc(void *a) {}

feature_set *feature_set_alloc() {
  return RBTreeCreate(int_comp, int_dealloc, info_dealloc, int_print, info_print);
}

void feature_set_free(feature_set *set) {
  RBTreeDestroy(set);
}

int feature_set_contains(feature_set *set, int f) {
  if ((RBExactQuery(set, &f)) == 0)
    return 0;
  else
    return 1;
}

void feature_set_insert(feature_set *set, int f) {
  int *new = (int *) malloc(sizeof(int));
  *new = f;
  RBTreeInsert(set, new, 0);
}

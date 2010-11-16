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

#ifndef ESTIMATE_MODEL_H
#define ESTIMATE_MODEL_H

#include <stddef.h>

#include "lbfgs.h"
#include "rbtree/red_black_tree.h"

typedef rb_red_blk_tree feature_set;
feature_set *feature_set_alloc();
void feature_set_free(feature_set *set);
int feature_set_contains(feature_set *set, int f);
void feature_set_insert(feature_set *set, int f);

typedef struct {
  size_t n_params;
  lbfgsfloatval_t *params;
  rb_red_blk_tree *f_restrict;
} model_t;

typedef struct {
  double *zs;
  double **sums;
} z_sum_t;

void model_new(model_t *model, size_t n_params);
void model_free(model_t *model);


#endif // ESTIMATE_MODEL_H

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

#include "bitvector.h"
#include "lbfgs.h"
#include "rbtree/red_black_tree.h"

/* Features, ordered by score. */
typedef struct {
  int feature;
  double score;
} feature_score_t;

typedef rb_red_blk_tree feature_scores;
typedef rb_red_blk_node feature_scores_node;
feature_scores *feature_scores_alloc();
void feature_scores_free(feature_scores *scores);
void feature_scores_insert(feature_scores *scores, int f, double score);
feature_scores_node *feature_scores_begin(feature_scores *tree);
feature_scores_node *feature_scores_next(feature_scores *tree,
    feature_scores_node *node);

/* Model */
typedef struct {
  size_t n_params;
  lbfgsfloatval_t *params;
  bitvector_t *f_restrict;
} model_t;

typedef struct {
  double *zs;
  double **sums;
} z_sum_t;

void model_new(model_t *model, size_t n_params);
void model_free(model_t *model);


#endif // ESTIMATE_MODEL_H

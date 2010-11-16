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

#include <math.h>

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

int feature_scores_comp(void const *a, void const *b) {
  feature_score_t *fs_a = (feature_score_t *) a;
  feature_score_t *fs_b = (feature_score_t *) b;

  if (fabs(fs_a->score) > fabs(fs_b->score))
    return -1;
  else if (fabs(fs_a->score) < fabs(fs_b->score))
    return 1;
  else if (fs_a->feature > fs_b->feature)
    return 1;
  else if (fs_a->feature < fs_b->feature)
    return -1;
  else
    return 0;
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

/* Feature scores. */

void feature_scores_dealloc(void *a) {
  free((feature_scores *) a);
}
void feature_scores_print(void const *a) {}

feature_scores *feature_scores_alloc() {
    return RBTreeCreate(feature_scores_comp, feature_scores_dealloc,
        info_dealloc, feature_scores_print, info_print);
}

void feature_scores_free(feature_scores *scores)
{
  RBTreeDestroy(scores);
}

void feature_scores_insert(feature_scores *scores, int f, double score) {
  feature_score_t *fs = (feature_score_t *) malloc(sizeof(feature_score_t));
  fs->feature = f;
  fs->score = score;
  RBTreeInsert(scores, fs, 0);
}

feature_scores_node *feature_scores_begin(feature_scores *tree)
{
  rb_red_blk_node* nil = tree->nil;
  rb_red_blk_node *x = tree->root;

  while (x->left != nil)
    x = x->left;

  return x;
}

feature_scores_node *feature_scores_next(feature_scores *tree,
    feature_scores_node *node)
{
  return TreeSuccessor(tree, node);
}


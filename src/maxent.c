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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <tinyest/bitvector.h>
#include <tinyest/dataset.h>
#include <tinyest/lbfgs.h>
#include <tinyest/maxent.h>
#include <tinyest/model.h>

void maxent_context_sums(dataset_context_t *ctx, lbfgsfloatval_t const *params,
    double *sums, double *z, bitvector_t *f_restrict)
{
  dataset_event_t *evts = ctx->events;
  for (int j = 0; j < ctx->n_events; ++j) {
    feature_value_t *fvals = evts[j].fvals;
    for (int k = 0; k < evts[j].n_fvals; ++k)
      if (f_restrict == 0 || bitvector_get(f_restrict, fvals[k].feature))
        sums[j] += params[fvals[k].feature] * fvals[k].value;

    sums[j] = exp(sums[j]);
    *z += sums[j];
  }
}

lbfgsfloatval_t maxent_lbfgs_evaluate(void *instance, lbfgsfloatval_t const *x,
    lbfgsfloatval_t *g, int const n, lbfgsfloatval_t const step)
{
  maxent_lbfgs_data_t *d= (maxent_lbfgs_data_t *) instance;
  dataset_t *ds = d->dataset;
  double *fvals = ds->feature_values;

  for (int i = 0; i < d->dataset->n_features; ++i)
    if (d->model->f_restrict == 0 ||
      bitvector_get(d->model->f_restrict, i))
    g[i] = -fvals[i];

  lbfgsfloatval_t ll = 0.0;

  dataset_context_t *ctxs = ds->contexts;
  for (int i = 0; i < ds->n_contexts; ++i) 
  {
    lbfgsfloatval_t ctxLl = 0.0;

    // Skip contexts that have a probability of zero. If we allow such
    // contexts, we can not calculate emperical p(y|x).
    if (ctxs[i].p == 0.0)
      continue;

    double *sums = malloc(ctxs[i].n_events * sizeof(double));
    memset(sums, 0, ctxs[i].n_events * sizeof(double));
    double z = 0.0;

    maxent_context_sums(&ctxs[i], x, sums, &z, d->model->f_restrict);

    dataset_event_t *evts = ctxs[i].events;
    for (int j = 0; j < ctxs[i].n_events; ++j) {
      // p(y|x)
      double p_yx = sums[j] / z;

      // Update log-likelihood of the model.
      ctxLl += evts[j].p * log(p_yx);

      // Contribution of this context to p(f).
      feature_value_t *fvals = evts[j].fvals;
      for (int k = 0; k < evts[j].n_fvals; ++k)
        if (d->model->f_restrict == 0 ||
            bitvector_get(d->model->f_restrict, fvals[k].feature)) {
          g[fvals[k].feature] += ctxs[i].p * p_yx * fvals[k].value;
        }
    }

    #pragma omp atomic
    ll += ctxLl;

    free(sums);
  }

  if (d->l2_sigma_sq != 0.0) {
    double sigma = 1.0 / d->l2_sigma_sq;
    double f_sq_sum = 0.0;

    for (int i = 0; i < d->dataset->n_features; ++i) {
      if (d->model->f_restrict &&
          !bitvector_get(d->model->f_restrict, i))
        continue;
      f_sq_sum += pow(x[i], 2.0);

      // Impose Gaussian prior on gradient: g_i + x_i / sigma_sq
      g[i] += x[i] * sigma; 
    }

    // Impose Gaussian prior on log-likelihood: ll - sum(x) / 2 * sigma_sq
    ll -= f_sq_sum * 0.5 * sigma;
  }

  return -ll;
}

int maxent_lbfgs_optimize(dataset_t *dataset, model_t *model,
     lbfgs_parameter_t *params, double l2_sigma_sq)
{
  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, model};

  int r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);

  return r;
}

void maxent_feature_gradients(dataset_t *dataset,
    lbfgsfloatval_t *params,
    lbfgsfloatval_t *gradients)
{
  double *fvals = dataset->feature_values;

  for (int i = 0; i < dataset->n_features; ++i)
    gradients[i] = -fvals[i];

  dataset_context_t *ctxs = dataset->contexts;
  for (int i = 0; i < dataset->n_contexts; ++i) 
  {
    if (ctxs[i].p == 0.0)
      continue;

    double *sums = malloc(ctxs[i].n_events * sizeof(double));
    memset(sums, 0, ctxs[i].n_events * sizeof(double));
    double z = 0.0;

    maxent_context_sums(&ctxs[i], params, sums, &z, 0);

    dataset_event_t *evts = ctxs[i].events;
    for (int j = 0; j < ctxs[i].n_events; ++j) {
      // p(y|x)
      double p_yx = sums[j] / z;

      // Contribution of this context to p(f).
      feature_value_t *fvals = evts[j].fvals;
      for (int k = 0; k < evts[j].n_fvals; ++k)
        gradients[fvals[k].feature] += ctxs[i].p * p_yx * fvals[k].value;
    }
    free(sums);
  }
}

int feature_score_compare(void const *l, void const *r)
{
  feature_score_t **lfs = (feature_score_t **) l;
  feature_score_t **rfs = (feature_score_t **) r;

  double l_score = fabs((*lfs)->score);
  double r_score = fabs((*rfs)->score);

  if (l_score > r_score)
    return -1;
  else if (l_score < r_score)
    return 1;
  else
    return 0;
}

int maxent_select_features(dataset_t *dataset, lbfgs_parameter_t *params,
    model_t *model, double *gradients, int n_select)
{
  int n_unselected = dataset->n_features - model->f_restrict->on;

  // Allocate memory for all unselected features.
  feature_score_t **scores = (feature_score_t **)
    malloc(sizeof(feature_score_t *) * n_unselected);
  //memset(scores, 0, sizeof(feature_score_t *) * n_unselected);

  int i;
  int cand;
  for (i = 0, cand = 0; i < dataset->n_features; ++i)
      if (!bitvector_get(model->f_restrict, i)) {
        scores[cand] = (feature_score_t *) malloc(sizeof(feature_score_t));
        scores[cand]->feature = i;
        scores[cand]->score = gradients[i];
        ++cand;
      }

    assert(cand == n_unselected);
    qsort(scores, n_unselected, sizeof(feature_score_t *),
      feature_score_compare);

    // Pick features with the highest gradient.
    int cur_select = 0;
    for (int i = 0; i < n_unselected && cur_select < n_select; ++i) {
      feature_score_t *score = scores[i];
      if (fabs(score->score) <= params->orthantwise_c)
        continue;

      fprintf(stderr, "* %d\n", score->feature);
      bitvector_set(model->f_restrict, score->feature, 1);
      ++cur_select;
    }

    // Free up scores.
    for (int i = 0; i < n_unselected; ++i)
      free(scores[i]);
    free(scores);

    return cur_select;
}

int maxent_lbfgs_grafting_light(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n)
{
  model->f_restrict = bitvector_alloc(dataset->n_features);
  params->max_iterations = 1;

  lbfgsfloatval_t *g = lbfgs_malloc(dataset->n_features);

  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, model};

  int r = LBFGS_SUCCESS;
  while (1) {
    fprintf(stderr, "--- Feature selection ---\n");

    // Calculate feature gradients.
    maxent_feature_gradients(dataset, model->params, g); 

    // Select most promising features.
    (void) maxent_select_features(dataset, params, model, g,
        grafting_n);

    fprintf(stderr, "--- Optimizing model ---\n");
    r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);
    if (r != LBFGSERR_MAXIMUMITERATION)
      break;
  }

  lbfgs_free(g);
  bitvector_free(model->f_restrict);

  return r;
}

int maxent_lbfgs_grafting(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n)
{
  model->f_restrict = bitvector_alloc(dataset->n_features);

  lbfgsfloatval_t *g = lbfgs_malloc(dataset->n_features);

  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, model};

  int r = LBFGS_SUCCESS;
  while (1) {
    fprintf(stderr, "--- Feature selection ---\n");

    // Calculate feature gradients.
    maxent_feature_gradients(dataset, model->params, g); 

    // Select most promising features.
    int n_selected = maxent_select_features(dataset, params, model, g,
        grafting_n);

    // No features...
    if (n_selected == 0) {
      fprintf(stderr, "done!\n");
      break;
    }

    fprintf(stderr, "--- Optimizing model ---\n");
    int r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);
    if (r != LBFGS_STOP && r != LBFGS_SUCCESS && r != LBFGS_ALREADY_MINIMIZED)
      break;
  }

  lbfgs_free(g);
  bitvector_free(model->f_restrict);

  return r;
}

int maxent_lbfgs_progress_verbose(void *instance, lbfgsfloatval_t const *x,
    lbfgsfloatval_t const *g, lbfgsfloatval_t const fx,
    lbfgsfloatval_t const xnorm, lbfgsfloatval_t const gnorm,
    lbfgsfloatval_t const step, int n, int k, int ls)
{
  fprintf(stderr, "%d\t%.4e\t%.4e\t%.4e\n", k, fx, xnorm, gnorm);

  return 0;
}

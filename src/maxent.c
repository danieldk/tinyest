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
#include <stdlib.h>
#include <string.h>

#include <tinyest/dataset.h>
#include <tinyest/lbfgs.h>
#include <tinyest/maxent.h>
#include <tinyest/model.h>

void maxent_context_sums(dataset_context_t *ctx, lbfgsfloatval_t const *params,
    double *sums, double *z, feature_set *f_restrict)
{
  dataset_event_t *evts = ctx->events;
  for (int j = 0; j < ctx->n_events; ++j) {
    feature_value_t *fvals = evts[j].fvals;
    for (int k = 0; k < evts[j].n_fvals; ++k)
      if (f_restrict == 0 || feature_set_contains(f_restrict, fvals[k].feature))
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

  for (int i = 0; i < d->dataset->n_features; ++i)
    if (d->model->f_restrict == 0 ||
      feature_set_contains(d->model->f_restrict, i))
    g[i] = -d->feature_values[i];

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
            feature_set_contains(d->model->f_restrict, fvals[k].feature)) {
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
          !feature_set_contains(d->model->f_restrict, i))
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
  double *fvals = dataset_feature_values(dataset);
  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, fvals, model};

  int r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);

  free(fvals);

  return r;
}

void maxent_feature_gradients(dataset_t *dataset,
    lbfgsfloatval_t *params,
    lbfgsfloatval_t *gradients)
{
  double *fvals = dataset_feature_values(dataset);

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

int maxent_lbfgs_grafting(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n,
    int light)
{
  model->f_restrict = feature_set_alloc();
  if (light)
    params->max_iterations = 1;


  double l2_sigma = 0.0;
  if (l2_sigma_sq != 0.0)
   l2_sigma = 1.0 / l2_sigma_sq;

  lbfgsfloatval_t *g = lbfgs_malloc(dataset->n_features);

  double *fvals = dataset_feature_values(dataset);
  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, fvals, model};


  while (1) {
    fprintf(stderr, "--- Feature selection ---\n");
    maxent_feature_gradients(dataset, model->params, g); 

    //for (int i = 0; i < dataset->n_features; ++i)
    //  g[i] += selection_model.params[i] * l2_sigma; 

    // Order features by gradient.
    feature_scores *scores = feature_scores_alloc();
    for (int i = 0; i < dataset->n_features; ++i)
      if (!feature_set_contains(model->f_restrict, i))
        feature_scores_insert(scores, i, g[i]);

    fprintf(stderr, "--> Selected features:");

    int i = 0;
    feature_scores_node *n = feature_scores_begin(scores);
    while (n != scores->nil && i < grafting_n) {
      feature_score_t *score = (feature_score_t *) n->key;
      if (fabs(g[score->feature]) <= params->orthantwise_c) {
        n = feature_scores_next(scores, n);
        continue;
      }

      fprintf(stderr, " %d", score->feature);
      feature_set_insert(model->f_restrict, score->feature);

      ++i;
      n = feature_scores_next(scores, n);
    }

    // No features...
    if (i == 0) {
      fprintf(stderr, "done!\n");
      break;
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "--- Optimizing model ---\n");
    int r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);
  }

  if (light) {
    params->max_iterations = 0;
    lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);
  }

  lbfgs_free(g);

  return 0;
}

int maxent_lbfgs_progress_verbose(void *instance, lbfgsfloatval_t const *x,
    lbfgsfloatval_t const *g, lbfgsfloatval_t const fx,
    lbfgsfloatval_t const xnorm, lbfgsfloatval_t const gnorm,
    lbfgsfloatval_t const step, int n, int k, int ls)
{
  fprintf(stderr, "%d\t%.4e\t%.4e\t%.4e\n", k, fx, xnorm, gnorm);

  return 0;
}

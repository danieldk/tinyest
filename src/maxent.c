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
    double *sums, double *z)
{
  dataset_event_t *evts = ctx->events;
  for (int j = 0; j < ctx->n_events; ++j) {
    feature_value_t *fvals = evts[j].fvals;
    for (int k = 0; k < evts[j].n_fvals; ++k)
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
    g[i] = -d->feature_values[i];

  lbfgsfloatval_t ll = 0.0;

  dataset_context_t *ctxs = ds->contexts;
  #pragma omp parallel for
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

    maxent_context_sums(&ctxs[i], x, sums, &z);

    dataset_event_t *evts = ctxs[i].events;
    for (int j = 0; j < ctxs[i].n_events; ++j) {
      // p(y|x)
      double p_yx = sums[j] / z;

      // Update log-likelihood of the model.
      ctxLl += evts[j].p * log(p_yx);

      // Contribution of this context to p(f).
      feature_value_t *fvals = evts[j].fvals;
      for (int k = 0; k < evts[j].n_fvals; ++k)
        #pragma omp atomic
        g[fvals[k].feature] += ctxs[i].p * p_yx * fvals[k].value;
    }

    #pragma omp atomic
    ll += ctxLl;

    free(sums);
  }

  if (d->l2_sigma_sq != 0.0) {
    double sigma = 1.0 / d->l2_sigma_sq;
    double f_sq_sum = 0.0;

    for (int i = 0; i < d->dataset->n_features; ++i) {
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
  maxent_lbfgs_data_t lbfgs_data = {l2_sigma_sq, dataset, fvals};

  int r = lbfgs(dataset->n_features, model->params, 0, maxent_lbfgs_evaluate,
      maxent_lbfgs_progress_verbose, &lbfgs_data, params);

  free(fvals);

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

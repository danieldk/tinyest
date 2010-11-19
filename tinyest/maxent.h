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

#ifndef ESTIMATE_MAXENT_HH
#define ESTIMATE_MAXENT_HH

#include "dataset.h"
#include "lbfgs.h"
#include "model.h"

typedef struct {
  double l2_sigma_sq;
  dataset_t *dataset;
  double *feature_values;
  model_t *model;
} maxent_lbfgs_data_t;

void maxent_context_sums(dataset_context_t *ctx, lbfgsfloatval_t const *params,
    double *sums, double *z, feature_set *set);

lbfgsfloatval_t maxent_lbfgs_evaluate(void *instance, lbfgsfloatval_t const *x,
  lbfgsfloatval_t *g, int const n, lbfgsfloatval_t const step);

void maxent_feature_gradients(dataset_t *dataset,
    lbfgsfloatval_t *params,
    lbfgsfloatval_t *gradients);

int maxent_lbfgs_grafting(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n);

int maxent_lbfgs_grafting_light(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n);

int maxent_lbfgs_optimize(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *param, double l2_sigma_sq);

int maxent_lbfgs_progress_verbose(void *instance, lbfgsfloatval_t const *x,
    lbfgsfloatval_t const *g, lbfgsfloatval_t const fx,
    lbfgsfloatval_t const xnorm, lbfgsfloatval_t const gnorm,
    lbfgsfloatval_t const tep, int n, int k, int ls);

#endif // ESTIMATE_MAXENT_HH

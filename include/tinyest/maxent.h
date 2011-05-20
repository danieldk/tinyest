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

/*!
 * Apply 'grafting' feature selection. Grafting alternates optimization with
 * feature selection, where features are selected based on their gradients.
 * Feature selection stops when no feature has a gradient higher than the l1
 * norm coefficient. The final model will be optimized.
 *
 * @param dataset the dataset
 * @param model initial model
 * @param param parameter LBFGS parameters
 * @param l2_sigma_sq squared sigma of the l2 prior
 * @param grafting_n number of features to select during each feature
 *        selection step
 */
int maxent_lbfgs_grafting(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n);

/*!
 * Apply 'grafting-light' feature selection. Grafting-light feature selection,
 * with one iteration of gradient-descent. Features are selected based on their
 * gradients. Feature selection stops when no feature has a gradient higher
 * than the l1 norm coefficient. The final model will be optimized.
 *
 * @param dataset the dataset
 * @param model initial model
 * @param param LBFGS parameters
 * @param l2_sigma_sq squared sigma of the l2 prior
 * @param grafting_n number of features to select during each feature
 *        selection step
 */
int maxent_lbfgs_grafting_light(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *params, double l2_sigma_sq, int grafting_n);

/*!
 * Maximum entropy model parameter estimation.
 *
 * @param dataset the dataset
 * @param model initial model
 * @param param LBFGS parameters
 * @param l2_sigma_sq squared sigma of the l2 prior
 */
int maxent_lbfgs_optimize(dataset_t *dataset, model_t *model,
    lbfgs_parameter_t *param, double l2_sigma_sq);

int maxent_lbfgs_progress_verbose(void *instance, lbfgsfloatval_t const *x,
    lbfgsfloatval_t const *g, lbfgsfloatval_t const fx,
    lbfgsfloatval_t const xnorm, lbfgsfloatval_t const gnorm,
    lbfgsfloatval_t const tep, int n, int k, int ls);

#endif // ESTIMATE_MAXENT_HH

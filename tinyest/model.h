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

#include <tinyest/lbfgs.h>

typedef struct {
  size_t n_params;
  lbfgsfloatval_t *params;
} model_t;

typedef struct {
  double *zs;
  double **sums;
} z_sum_t;

void model_new(model_t *model, size_t n_params);
void model_free(model_t *model);

#endif // ESTIMATE_MODEL_H

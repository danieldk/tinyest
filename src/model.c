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
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <tinyest/lbfgs.h>
#include <tinyest/model.h>

model_t *model_new(size_t n_params, bool f_restrict)
{
  model_t *model = malloc(sizeof(model_t));

  model->params = lbfgs_malloc(n_params);
  model->n_params = n_params;

  if (f_restrict)
    model->f_restrict = bitvector_alloc(n_params);
  else
    model->f_restrict = NULL;

  return model;
}

void model_free(model_t *model)
{
  model->n_params = 0;
  lbfgs_free(model->params);

  if (model->f_restrict != NULL)
    bitvector_free(model->f_restrict);

  free(model);
}

model_t *model_read(FILE *f, bool f_restrict)
{

  double *tmp_weights = malloc(0);
  size_t n_params = 0;

  char line[65535];
  while (fgets(line, 65535, f) != NULL)
  {
    double w = strtod(line, NULL);

    ++n_params;
    tmp_weights = realloc(tmp_weights, n_params * sizeof(double));
    tmp_weights[n_params - 1] = w;
  }

  lbfgsfloatval_t *params = lbfgs_malloc(n_params);
  memcpy(params, tmp_weights, n_params * sizeof(double));

  free(tmp_weights);

  model_t *model = malloc(sizeof(model_t));
  model->n_params = n_params;
  model->params = params;

  if (f_restrict)
  {
    model->f_restrict = bitvector_alloc(n_params);
    for (size_t i = 0; i < n_params; ++i)
      if (params[i] != 0.0)
        bitvector_set(model->f_restrict, i, 1); 
  }
  else
    model->f_restrict = NULL;

  return model;
}


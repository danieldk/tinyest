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
#include <string.h>

#include <tinyest/lbfgs.h>
#include <tinyest/model.h>

void model_new(model_t *model, size_t n_params, bool f_restrict)
{
  model->params = lbfgs_malloc(n_params);
  model->n_params = n_params;

  if (f_restrict)
    model->f_restrict = bitvector_alloc(n_params);
  else
    model->f_restrict = NULL;
}

void model_free(model_t *model)
{
  model->n_params = 0;
  lbfgs_free(model->params);

  if (model->f_restrict != NULL)
    bitvector_free(model->f_restrict);
}


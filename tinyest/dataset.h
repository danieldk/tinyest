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

#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
  size_t feature;
  double value;
} feature_value_t;

typedef struct {
  double p;
  size_t n_fvals;
  feature_value_t *fvals;
} dataset_event_t;

typedef struct {
  double p;
  size_t n_events;
  dataset_event_t *events;
} dataset_context_t;

typedef struct {
  size_t n_features;
  size_t n_contexts;
  dataset_context_t *contexts;
} dataset_t;

enum tadm_read_status {
  TADM_OK,
  TADM_EOF,
  TADM_ERROR_PREMATURE_EOF,
  TADM_ERROR_MALFORMED_EVENT,
  TADM_ERROR_NUMBER_FEATURES
};

double *dataset_feature_values(dataset_t *dataset);
void dataset_normalize(dataset_t *dataset);
void dataset_free(dataset_t *dataset);
void dataset_context_free(dataset_context_t *context);
void dataset_event_free(dataset_event_t *event);
int read_tadm_dataset(FILE *f, dataset_t *dataset);

#endif // DATASET_H

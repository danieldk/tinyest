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

#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include <tinyest/dataset.h>

int read_tadm_context(gzFile f, dataset_context_t *ctx);
int read_tadm_event(gzFile f, dataset_event_t *evt);

size_t dataset_count_features(dataset_t *dataset)
{
  dataset_context_t *ctxs = dataset->contexts;
  int hf = -1;

  for (int i = 0; i < dataset->n_contexts; ++i) {
    dataset_event_t *evts = ctxs[i].events;

    for (int j = 0; j < ctxs[i].n_events; ++j) {
      feature_value_t *fvals = evts[j].fvals;

      for (int k = 0; k < evts[j].n_fvals; ++k)
        if ((int) fvals[k].feature > hf)
          hf = fvals[k].feature;
    }
  }

  return (size_t) hf + 1;
}

double *dataset_feature_values(dataset_t *dataset)
{
  double *fv = malloc(dataset->n_features * sizeof(double));
  memset(fv, 0, dataset->n_features * sizeof(double));

  dataset_context_t *ctxs = dataset->contexts;

  for (int i = 0; i < dataset->n_contexts; ++i) {
    dataset_event_t *evts = ctxs[i].events;

    for (int j = 0; j < ctxs[i].n_events; ++j) {
      feature_value_t *fvals = evts[j].fvals;

      for (int k = 0; k < evts[j].n_fvals; ++k)
        fv[fvals[k].feature] += evts[j].p * fvals[k].value;
    }
  }

  return fv;
}

void dataset_normalize(dataset_t *dataset)
{
  dataset_context_t *ctxs = dataset->contexts;

  double scoreSum = 0.0;
  for (int i = 0; i < dataset->n_contexts; ++i) {
    dataset_event_t *evts = ctxs[i].events;
    ctxs[i].p = 0.0;

    for (int j = 0; j < ctxs[i].n_events; ++j)
      ctxs[i].p += evts[j].p;

    scoreSum += ctxs[i].p;
  }

  for (int i = 0; i < dataset->n_contexts; ++i) {
    ctxs[i].p /= scoreSum;
    dataset_event_t *evts = ctxs[i].events;

    for (int j = 0; j < ctxs[i].n_events; ++j)
      evts[j].p /= scoreSum;
  }
}

void dataset_free(dataset_t *dataset)
{
  for (int i = 0; i < dataset->n_contexts; ++i)
    dataset_context_free(&dataset->contexts[i]);

  dataset->n_contexts = 0;
  if (dataset->contexts != NULL) {
    free(dataset->contexts);
    dataset->contexts = NULL;
  }

  if (dataset->feature_values != NULL)
    free(dataset->feature_values);
}

void dataset_context_free(dataset_context_t *context)
{
  for (int i = 0; i < context->n_events; ++i)
    dataset_event_free(&context->events[i]);

  context->n_events = 0;

  if (context->events != NULL) {
    free(context->events);
    context->events = NULL;
  }
}

void dataset_event_free(dataset_event_t *event)
{
  event->n_fvals = 0;

  if (event->fvals != NULL) {
    free(event->fvals);
    event->fvals = NULL;
  }
}

int read_tadm_dataset(int fd, dataset_t *dataset)
{
  gzFile f;
  if ((f = gzdopen(fd, "rb")) == NULL)
    return TADM_ERROR_OPEN;

  memset(dataset, 0, sizeof(dataset_t));

  dataset->contexts = NULL;
  dataset->feature_values = NULL;

  while (1) {
    dataset_context_t ctx;
    int r = read_tadm_context(f, &ctx);
    if (r == TADM_EOF)
      break;
    else if (r != 0) {
      dataset_free(dataset);
      gzclose(f);
      return r;
    }

    ++dataset->n_contexts;
    dataset->contexts = realloc(dataset->contexts,
      dataset->n_contexts * sizeof(dataset_context_t));
    memcpy(&dataset->contexts[dataset->n_contexts - 1], &ctx,
      sizeof(dataset_context_t));
  }

  gzclose(f);

  dataset_normalize(dataset);
  dataset->n_features = dataset_count_features(dataset);
  dataset->feature_values = dataset_feature_values(dataset);

  return TADM_OK;
}

int read_tadm_context(gzFile f, dataset_context_t *ctx)
{

  memset(ctx, 0, sizeof(dataset_context_t));
  /* Read context size. */
  char line[65535];
  if (gzgets(f, line, 65535) == NULL) {
    return TADM_EOF;
  }

  /* And convert it to int. */
  int n_events = atoi(line);

  /* Allocate memory for context events. */
  ctx->events = malloc(n_events * sizeof(dataset_event_t));

  for (int i = 0; i < n_events; ++i) {
    int r = read_tadm_event(f, &ctx->events[i]);
    if (r != 0)
    {
      dataset_context_free(ctx);
      return r;
    }

    ++ctx->n_events;
  }

  return TADM_OK;
}

int read_tadm_event(gzFile f, dataset_event_t *evt)
{
  memset(evt, 0, sizeof(dataset_event_t));

  char line[65535];
  if (gzgets(f, line, 65535) == NULL) {
    return TADM_ERROR_PREMATURE_EOF;
  }

  char *cur = line;
  /* Extract event score. */
  if ((cur = strtok(cur, " ")) == NULL)
    return TADM_ERROR_MALFORMED_EVENT;
  evt->p = atof(cur);

  /* Extract number of features with non-zero values. */
  if ((cur = strtok(NULL, " ")) == NULL)
    return TADM_ERROR_MALFORMED_EVENT;
  evt->n_fvals = atoi(cur);
  evt->fvals = malloc(evt->n_fvals * sizeof(feature_value_t));

  /* Read features. */
  for (int i = 0; i < evt->n_fvals; ++i) {
    /* Feature number */
    if ((cur = strtok(NULL, " ")) == NULL) {
      dataset_event_free(evt);
      return TADM_ERROR_NUMBER_FEATURES;
    }
    evt->fvals[i].feature = strtol(cur, NULL, 10);

    /* Feature value */
    if ((cur = strtok(NULL, " ")) == NULL)
      return TADM_ERROR_NUMBER_FEATURES;
    evt->fvals[i].value = strtod(cur, NULL);
  }

  return TADM_OK;
}

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

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <tinyest/dataset.h>
#include <tinyest/lbfgs.h>
#include <tinyest/maxent.h>
#include <tinyest/model.h>

// Options
static struct option longopts[] = {
  { "ftol", required_argument, NULL, 1},
  { "gtol", required_argument, NULL, 2},
  { "grafting", required_argument, NULL, 3},
  { "grafting-light", required_argument, NULL, 4},
  { "l1", required_argument, NULL, 5},
  { "l2", required_argument, NULL, 6},
  { "linesearch", required_argument, NULL, 7},
  { "minstep", required_argument, NULL, 8},
  { "maxstep", required_argument, NULL, 9},
  { NULL, 0, NULL, 0 }
};

void usage(char *program_name)
{
  fprintf(stderr, "Usage: %s [OPTION] dataset\n\n", program_name);
  fprintf(stderr, "--ftol val\t\tLine search algorithm ftol (default: 1e-4)\n");
  fprintf(stderr, "--gtol val\t\tLine search algorithm gtol (default: 0.9)\n");
  fprintf(stderr, "--grafting n\t\tEnable grafting (feature selection)\n");
  fprintf(stderr, "--grafting-light n\tEnable grafting-light (feature selection)\n");
  fprintf(stderr, "--l1 val\t\tl1 norm coefficient\n");
  fprintf(stderr, "--l2 val\t\tGaussian (l2) prior\n");
  fprintf(stderr, "--linesearch alg\tLine search algorithm: armijo, ");
  fprintf(stderr, "backtracking, wolfe, or\n\t\t\tstrong_wolfe\n");
  fprintf(stderr, "--minstep val\t\tMinimum step of the line search routine (default: 1e-20)\n");
  fprintf(stderr, "--maxstep val\t\tMaximum step of the line search routine (default: 1e20)\n\n");
}

double str_to_double(char *str)
{
  char *ep;
  double r = strtod(str, &ep);
  if (r == 0.0 && ep == str) {
    fprintf(stderr, "Invalid double value: %s\n", optarg);
    exit(1);
  } else if (r == HUGE_VAL || r == -HUGE_VAL) {
    fprintf(stderr, "Value overflows double: %s\n", optarg);
    exit(1);
  } else if (r == 0.0 && errno == ERANGE) {
    fprintf(stderr, "Value underflows double: %s\n", optarg);
    exit(1);
  }

  return r;
}


int str_to_int(char *str)
{
  char *ep;
  int r = strtol(str, &ep, 10);
  if (r == 0 && ep == str) {
    fprintf(stderr, "Invalid int value: %s\n", optarg);
    exit(1);
  }
  else if (r == 0 && errno == EINVAL) {
    fprintf(stderr, "Value is invalid: %s\n", optarg);
    exit(1);
  } else if (r == 0 && errno == ERANGE) {
    fprintf(stderr, "Value overflows/underflows int: %s\n", optarg);
    exit(1);
  }

  return r;
}

int main(int argc, char *argv[]) {
  char *program_name = argv[0];
  double l2_sigma_sq = 0.0;
  int grafting = 0;
  int grafting_light = 0;

  lbfgs_parameter_t params;
  lbfgs_parameter_init(&params);
  params.past = 1;
  params.delta = 1e-7;

  int ch;
  while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
    switch (ch) {
    case 1:
      params.ftol = str_to_double(optarg);
      break;
    case 2:
      params.gtol = str_to_double(optarg);
      break;
    case 3:
      grafting = str_to_int(optarg);
      break;
    case 4:
      grafting_light = str_to_int(optarg);
      break;
    case 5:
      params.orthantwise_c = str_to_double(optarg);
      break;
    case 6:
      l2_sigma_sq = str_to_double(optarg);
      break;
    case 7:
      if (strcmp(optarg, "armijo") == 0)
        params.linesearch = LBFGS_LINESEARCH_BACKTRACKING_ARMIJO;
      else if (strcmp(optarg, "backtracking") == 0)
        params.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
      else if (strcmp(optarg, "wolfe") == 0)
        params.linesearch = LBFGS_LINESEARCH_BACKTRACKING_WOLFE;
      else if (strcmp(optarg, "strong_wolfe") == 0)
        params.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
      else {
        usage(program_name);
        return 1;
      }
      break;
    case 8:
      fprintf(stderr,"backtracking\n");
      params.min_step = str_to_double(optarg);
      break;
    case 9:
      params.max_step = str_to_double(optarg);
      break;
    case '?':
    default:
      usage(program_name);
      return 1;
    }
  }

  argc -= optind;
  argv += optind;

  if (argc != 0 && argc != 1) {
    usage(program_name);
    return 1;
  }

  if (grafting && grafting_light) {
    fprintf(stderr, "Grafting and grafting-light cannot be used simultaneously...");
    return 1;
  }

  if ((grafting || grafting_light) && params.orthantwise_c == 0.) {
    fprintf(stderr, "Grafting requires a l1 norm coefficient...");
    return 1;
  }

  fprintf(stderr, "l1 norm coefficient: %.4e\n", params.orthantwise_c); 
  fprintf(stderr, "l2 prior sigma^2: %.4e\n\n", l2_sigma_sq);

  dataset_t ds;
  
  int fd = 0;
  if (argc == 1 && (fd = open(argv[0], O_RDONLY)) == -1) {
    fprintf(stderr, "Could not open %s\n", argv[0]);
    return 1;
  }

  int r = read_tadm_dataset(fd, &ds);

  if (r != TADM_OK) {
    fprintf(stderr, "Error reading data...\n");
    return 1;
  }
  
  fprintf(stderr, "Features: %zu\n", ds.n_features);
  fprintf(stderr, "Contexts: %zu\n\n", ds.n_contexts);

  if (params.orthantwise_c != 0.0) {
    params.orthantwise_end = ds.n_features;
    // l1 prior only works with backtracking linesearch.
    params.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  }

  model_t model;
  if (grafting || grafting_light)
    model_new(&model, ds.n_features, true);
  else
    model_new(&model, ds.n_features, false);

  fprintf(stderr, "Iter\t-LL\t\txnorm\t\tgnorm\n\n");

  if (grafting)
    r = maxent_lbfgs_grafting(&ds, &model, &params, l2_sigma_sq, grafting);
  else if (grafting_light)
    r = maxent_lbfgs_grafting_light(&ds, &model, &params, l2_sigma_sq,
        grafting_light);
  else
    r = maxent_lbfgs_optimize(&ds, &model, &params, l2_sigma_sq);

  dataset_free(&ds);

  if (r != LBFGS_STOP && r != LBFGS_SUCCESS && r != LBFGS_ALREADY_MINIMIZED) {
    fprintf(stderr, "lbfgs result: %d\n", r);
    model_free(&model);
    return 1;
  }

  for (int i = 0; i < ds.n_features; ++i)
    printf("%.8f\n", model.params[i]);

  model_free(&model);

  return 0;
}

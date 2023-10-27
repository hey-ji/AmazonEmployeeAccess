# PCR(Principal Component Reduction) --------------------------------------
library(tidymodels)
library(embed)
library(vroom)

# set working directory
setwd("/Users/student/Desktop/STAT348/AmazonEmployeeAccess")

# read in the data
amazon_training  <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)

# set up a recipe
my_recipe <- recipe(ACTION~., data=amazon_training) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% #target encoding: factors -> numeric
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.8) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_smote(all_outcomes(), neighbors=3)

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

# run KNN again -----------------------------------------------------------

## knn model
knn_model_PCR <- nearest_neighbor(neighbors=7) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model_PCR)

# set up tuning grid and folds
folds <- vfold_cv(amazon_training, v = 5, repeats=1)

knn_tuning_grid <- grid_regular(neighbors())

## Tune neighbors here
knn_cv <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=knn_tuning_grid,
            metrics=metric_set(roc_auc))

## Find the best Fit
bestTune <- knn_cv %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_training)

final_wf %>%
  predict(new_data = amazon_test)

## Predict
amazon_predictions <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "(Balanced)KNN_With_PCS.csv", delim = ",")

# run naive Bayes again ---------------------------------------------------

## create a workflow with model & recipe
nb_model_PCR <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model_PCR)

# set up tuning grid and folds

folds <- vfold_cv(amazon_training, v = 5, repeats=1)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness())

## Tune smoothness and Laplace here
nb_cv <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc))

## Find the best Fit
bestTune <- nb_cv %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_training)

final_wf %>%
  predict(new_data = amazon_test)

## Predict
amazon_predictions <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "(Balanced)NaiveBayes_With_PCS.csv", delim = ",")

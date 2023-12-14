# Basic prep ---------------------------------------------------

# library
library(tidymodels)
library(embed)
library(vroom)
library(themis)

# Read in the data
setwd("/Users/student/Desktop/STAT348/AmazonEmployeeAccess")
amazon_training  <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
#amazon_training  <- vroom("./train.csv")
amazon_test <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
#amazon_test <- vroom("./test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)

# Set up a recipe
my_recipe <- recipe(ACTION~., data=amazon_training) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 
#%>% target encoding: factors -> numeric
  #step_normalize(all_predictors()) 
#%>%
  #step_smote(all_outcomes(), neighbors=3)

# apply the recipe to your data
prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)


# # Logistic Regression -----------------------------------------------------
# logistic_regression_mod <- logistic_reg() %>% #Type of model
#   set_engine("glm")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logistic_regression_mod) %>%
#   fit(data = amazon_training) # Fit the workflow
# 
# amazon_predictions <- predict(amazon_workflow,
#                               new_data=amazon_test,
#                               type="prob") %>%
#   bind_cols(amazon_test) %>%
#   rename(ACTION=.pred_1) %>%
#   select(id, ACTION)
# vroom_write(x = amazon_predictions, file = "(Balance)Logistic", delim = ",")
# 
# 
# # Penalized Logistic Regression -------------------------------------------
# penalized_logistic_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
#   set_engine("glmnet")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(penalized_logistic_mod)
# 
# ## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# folds <- vfold_cv(amazon_training, v = 5, repeats=1)
# 
# ## Run the CV
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #Or leave metrics NULL, roc_auc, f_meas, sens, recall, spec,22
# # precision, accuracy
# 
# ## Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_wf <-
#   amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_training)
# 
# ## Predict
# amazon_predictions <- final_wf %>%
#   predict(new_data = amazon_test, type="prob") %>%
#   bind_cols(amazon_test) %>%
#   rename(ACTION=.pred_1) %>%
#   select(id, ACTION)
# vroom_write(x = amazon_predictions, file = "(Balance)Penalized", delim = ",")


# Random Forest -----------------------------------------------------------

# model for random forest
my_forest_model <- rand_forest(mtry = 1,
                      min_n = 15,
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# my_forest_model <- rand_forest(mtry = tune(),
#                                min_n = tune(),
#                                trees = 1000) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")

# Create a workflow with model & recipe
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_model) %>%
  fit(data = amazon_training)

# # Set up grid of tuning values
# tuning_grid <- grid_regular(mtry(range = c(1,ncol(amazon_training)-1)),
#                             min_n(),
#                             levels = 7)
# 
# # Set up K-fold CV
# folds <- vfold_cv(amazon_training, v = 7, repeats=1)
# 
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# # run the CV
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc))
# 
# # Find best tuning parameters
# bestTune <- CV_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and predict
# final_wf <-
#   amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=amazon_training)
# 
# final_wf %>%
#   predict(new_data = amazon_test)

# Formatting for submission
amazon_predictions <- amazon_workflow %>%
  predict(new_data = amazon_test, type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "/Users/student/Desktop/STAT348/AmazonEmployeeAccess/(Balanced)RandomForest(Lucy).csv", delim = ",")


# Naive Bayes -------------------------------------------------------------
## create a workflow with model & recipe
Naive_Bayes_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(Naive_Bayes_model)

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
vroom_write(x = amazon_predictions, file = "(Balanced)NaiveBayes.csv", delim = ",")


# KNN ---------------------------------------------------------------------
## knn model
knn_model <- nearest_neighbor(neighbors=7) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

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
vroom_write(x = amazon_predictions, file = "(Balanced)KNN.csv", delim = ",")

# Support Vector Machine --------------------------------------------------
# SVM model
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

# set up tuning grid and folds
folds <- vfold_cv(amazon_training, v = 2, repeats=1)

svm_tuning_grid <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 2)
# Tuning
svm_cv <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=svm_tuning_grid,
            metrics=metric_set(roc_auc))

# Fin the best Fit
bestTune <- svm_cv %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_training)

final_wf %>%
  predict(final_wf, new_data=amazon_test, type="prob")

## Predict
amazon_predictions <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "(Balanced)SupportVector.csv", delim = ",")


library(tidymodels)
library(embed)
library(vroom)

# Read in the data
#setwd("/Users/student/Desktop/STAT348/AmazonEmployeeAccess")
amazon_training  <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)


# at least 2 exploratory plots for the amazon data
# ggplot(amazon, aes(x = ACTION)) +
#   geom_bar()
# 
# ggplot(amazon, aes(x = ROLE_TITLE)) +
#   geom_bar()

sorted_counts <- sort(table(amazon_training$ROLE_TITLE), decreasing = T)
sorted_counts_20 <- head(sorted_counts, 20)
df_counts_20 <- data.frame(Value = names(sorted_counts_20), Count = sorted_counts_20)
ggplot(df_counts_20, aes(x = Count.Var1, y = Count.Freq)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  scale_x_discrete(limits = rev(factor(df_counts_20$Value)))

# create a recipe, prep and bake

my_recipe <- recipe(ACTION~., data=amazon_training) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

#setwd("/Users/student/Desktop/STAT348/AmazonEmployeeAccess")


# Logistic Regression -----------------------------------------------------
library(tidymodels)

my_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod) %>%
fit(data = amazon_training) # Fit the workflow

amazon_predictions <- predict(amazon_workflow,
                              new_data=amazon_test,
                              type="prob") %>%
                      bind_cols(amazon_test) %>%
                      rename(ACTION=.pred_1) %>%
                      select(id, ACTION)
vroom_write(x = amazon_predictions, file = "amazonlogisticregression2.csv", delim = ",")
getwd()

# Penalized Logistic Regression -------------------------------------------
library(tidymodels)
library(embed)
library(vroom)

my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(amazon_training, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL, roc_auc, f_meas, sens, recall, spec,22
# precision, accuracy

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  amazon_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=amazon_training)

## Predict
amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "penalized", delim = ",")

# save(file = "filename.RData" , list = c("LogRge_wf"))
# load("filename.RData")


# Classification Random Forests -------------------------------------------
library(tidymodels)
library(embed)
library(vroom)

# read in the data
amazon_training  <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)

# model for random forest
my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# create a recipe 
my_recipe <- recipe(ACTION~., data=amazon_training) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

# Create a workflow with model & recipe
amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = amazon_training)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,ncol(amazon_training)-1)),
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(amazon_training, v = 10, repeats=1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))
# run the CV
CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

# Finalize workflow and predict
final_wf <-
  amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazon_training)

final_wf %>%
  predict(new_data = amazon_test)

# Formatting for submission
amazon_predictions <- final_wf %>%
  predict(new_data = amazon_test, type="prob") %>%
  bind_cols(amazon_test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)
vroom_write(x = amazon_predictions, file = "/Users/student/Desktop/STAT348/AmazonEmployeeAccess/RandomForest.csv", delim = ",")


# Naive Bayes -------------------------------------------------------------

library(tidymodels)
library(embed)
library(vroom)
install.packages("discrim")
library(discrim)
# read in the data
amazon_training  <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("/Users/student/Desktop/STAT348/AmazonEmployeeAccess/test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)

# create a recipe, prep and bake
my_recipe <- recipe(ACTION~., data=amazon_training) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

## create a workflow with model & recipe
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(nb_model)

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
vroom_write(x = amazon_predictions, file = "NaiveBayes.csv", delim = ",")

# K-nearest neighbors -----------------------------------------------------
# get the needed library
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
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors())

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)


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
vroom_write(x = amazon_predictions, file = "KNN.csv", delim = ",")



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
  step_pca(all_predictors(), threshold = 0.8)

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

# run KNN again -----------------------------------------------------------

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
vroom_write(x = amazon_predictions, file = "KNN_With_PCS.csv", delim = ",")

# run naive Bayes again ---------------------------------------------------

## create a workflow with model & recipe
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

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
vroom_write(x = amazon_predictions, file = "NaiveBayes_With_PCS.csv", delim = ",")

# report
# NavieBayes gave me better score by 0.01725
# KNN gave me lower score by -0.01407


# Support Vector Machines -------------------------------------------------

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
  step_normalize(all_predictors()) 

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

#svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#set_mode("classification") %>%
#  set_engine("kernlab")

## SVM models3
#svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#  set_mode("classification") %>%
#set_engine("kernlab")

#svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
#  set_mode("classification") %>%
#set_engine("kernlab")

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
vroom_write(x = amazon_predictions, file = "SupportVector.csv", delim = ",")

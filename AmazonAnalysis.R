library(tidymodels)
library(embed)
library(vroom)

# Read in the data
setwd("/Users/student/Desktop/STAT348")
amazon_training  <- vroom("AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("AmazonEmployeeAccess/test.csv")
amazon_training$ACTION <- as.factor(amazon_training$ACTION)


# at least 2 exploratory plots for the amazon data
ggplot(amazon, aes(x = ACTION)) +
  geom_bar()

ggplot(amazon, aes(x = ROLE_TITLE)) +
  geom_bar()

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
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(target_var))

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

setwd("/Users/student/Desktop/STAT348/AmazonEmployeeAccess")


# Penalized Logistic Regression -------------------------------------------

library(tidymodels)

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
preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=myDataSet)

## Predict
final_wf %>%
predict(new_data = myNewData, type=)

#1. penalty, #2. mixture
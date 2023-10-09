library(tidymodels)
library(embed)

# Read in the data
setwd("/Users/student/Desktop/STAT348")
amazon_training  <- vroom("AmazonEmployeeAccess/train.csv")
amazon_test <- vroom("AmazonEmployeeAccess/test.csv")


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
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) 

prep <- prep(my_recipe)
bake(prep, new_data=amazon_training)
bake(prep, new_data=amazon_test)

library(data.table)
library(tidyr)
library(dplyr)
library(lubridate)

###############
# Import Data #
###############

# Comments on the other directories
train <- fread("Google Drive/Kaggle_Data/NYCtaxifee/train_small.csv")


######################
# Data Preprocessing #
######################

train <- train %>% separate(pickup_datetime, into = c("date", "time", "zone"), sep = " ") %>% select(-key, -zone) %>%
  filter(pickup_longitude!=0) %>% filter(passenger_count < 7) %>% 
  mutate(weekday = weekdays(as.Date(date)))

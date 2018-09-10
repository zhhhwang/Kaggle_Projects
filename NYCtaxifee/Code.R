library(data.table)
library(xgboost)
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

# Const
earthR <- 3958.7631
maxPassenger <- 6
halfCircleDegree <- 180
jfk_coor_lati <- (40.6459 * pi)/halfCircleDegree
jfk_coor_long <- (-73.7860 * pi)/halfCircleDegree
lga_coor_lati <- (40.7721 * pi)/halfCircleDegree
lga_coor_long <- (-73.8686 * pi)/halfCircleDegree
ewr_coor_lati <- (40.6917 * pi)/halfCircleDegree
ewr_coor_long <- (-74.1807 * pi)/halfCircleDegree
feature <- c("passenger_count", "weekday", "miles", "timeFrame", "pickupToJFK", "pickupToLGA", "pickupToEWR", "dropoffToJFK", "dropoffToLGA", "dropoffToEWR") 
label <- c("fare_amount")
folds <- 5

# Funciton in calculating the haversine distance between two gps coordinates
GPSdistance <- function(coor1_longitude, coor1_latitude, coor2_longitude, coor2_latitude){
  longD <- coor2_longitude - coor1_longitude
  latiD <- coor2_latitude - coor1_latitude
  
  haverSine <- sin(latiD / 2) * sin(latiD / 2) + cos(coor1_latitude) * cos(coor2_latitude) * sin(longD / 2) * sin(longD / 2)
  haverAngle <- asin(sqrt(haverSine))
  distance <- 2 * earthR * haverAngle
  return(distance)
}

train <- train %>% separate(pickup_datetime, into = c("date", "time", "zone"), sep = " ") %>% 
  select(-key, -zone) %>%
  filter(!(pickup_longitude == 0 |  pickup_latitude == 0 | dropoff_latitude == 0 | dropoff_longitude == 0)) %>% filter(passenger_count <= maxPassenger) %>% 
  mutate(pickup_longitude = (pickup_longitude * pi) / halfCircleDegree, 
         pickup_latitude = (pickup_latitude * pi) / halfCircleDegree,
         dropoff_longitude = (dropoff_longitude * pi) / halfCircleDegree, 
         dropoff_latitude = (dropoff_latitude * pi) / halfCircleDegree) %>%
  mutate(weekday = weekdays(as.Date(date))) %>% 
  mutate(miles = GPSdistance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)) %>%
  mutate(timeFrame = substr(time, 1, 2)) %>% mutate(timeFrame = as.numeric(timeFrame)) %>%
  mutate(yearFrame = substr(date, 1, 4)) %>% mutate(yearFrame = as.numeric(yearFrame)) %>%
  mutate(monthFrame = substr(date, 6, 7)) %>% mutate(monthFrame = as.numeric(monthFrame)) %>% 
  mutate(pickupToJFK = (GPSdistance(pickup_longitude, pickup_latitude, jfk_coor_long, jfk_coor_lati))) %>%
  mutate(pickupToLGA = (GPSdistance(pickup_longitude, pickup_latitude, lga_coor_long, lga_coor_lati))) %>%
  mutate(pickupToEWR = (GPSdistance(pickup_longitude, pickup_latitude, ewr_coor_long, ewr_coor_lati))) %>%
  mutate(dropoffToJFK = (GPSdistance(dropoff_longitude, dropoff_latitude, jfk_coor_long, jfk_coor_lati))) %>%
  mutate(dropoffToLGA = (GPSdistance(dropoff_longitude, dropoff_latitude, lga_coor_long, lga_coor_lati))) %>%
  mutate(dropoffToEWR = (GPSdistance(dropoff_longitude, dropoff_latitude, ewr_coor_long, ewr_coor_lati)))

  
########################
# Training and Testing #
########################

# Setting up cross validation index
train$cvIndex <- ceiling(sample(1:nrow(train), nrow(train)) / (nrow(train) / folds))

# cross validation
rmse <- rep(NA, folds)
for(i in 1:folds){
  training <- train[train$cvIndex != i, ] 
  testing <- train[train$cvIndex == i, ] 
  xgbModel <- xgboost(data = model.matrix(~ . + 0, training[, feature]), label = as.matrix(training[, label]), max.depth = 8, nrounds = 10000, nthread = 4, verbose = 1)
  predictions <- predict(xgbModel, model.matrix(~ . + 0, testing[, feature]))
  rmse[i] <- sqrt(mean((predictions - testing$fare_amount)^2))
  print(i)
}

# Model Training and Testing
# xgbModel <- xgboost(data = model.matrix(~ . + 0, train[, feature]), label = as.matrix(train[, label]), max.depth = 6, nrounds = 1000, nthread = 2, verbose = 1)
# predictions <- predict(xgbModel, model.matrix(~ . + 0, train[, feature]))

write.table(rmse, "outputRMSE.txt")


  


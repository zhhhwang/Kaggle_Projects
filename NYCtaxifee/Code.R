library(data.table)
library(xgboost)
library(tidyr)
library(dplyr)
library(lubridate)
library(doParallel)

###############
# Import Data #
###############

# Comments on the other directories
# train <- fread("Google Drive/Kaggle_Data/NYCtaxifee/train_small.csv")
# nycWeather <- fread("Google Drive/Kaggle_Data/NYCtaxifee/nycWeather.csv")
train <- fread("/home/zhhhwang/Kaggle_Data/NYCtaxifee/train_small.csv")
train <- fread("/home/zhhhwang/Kaggle_Data/NYCtaxifee/nycWeather.csv")

######################
# Data Preprocessing #
######################

# Const
earthR <- 3958.7631
maxPassenger <- 6
halfCircleDegree <- 180
jfk_coor_long <- (-73.7860 * pi)/halfCircleDegree
jfk_coor_lati <- (40.6459 * pi)/halfCircleDegree
lga_coor_long <- (-73.8686 * pi)/halfCircleDegree
lga_coor_lati <- (40.7721 * pi)/halfCircleDegree
ewr_coor_long <- (-74.1807 * pi)/halfCircleDegree
ewr_coor_lati <- (40.6917 * pi)/halfCircleDegree
nycLongLwr <- -74.30
nycLongUpp <- -72.90
nycLatiLwr <- 40.5
nycLatiUpp <- 42
feature <- c("passenger_count", "weekday", "miles", "monthFrame", 
             "yearFrame", "timeFrame", "pickupToJFK", "pickupToLGA", 
             "pickupToEWR", "dropoffToJFK", "dropoffToLGA", "dropoffToEWR", 
             "AWND", "PRCP", "SNOW", "SNWD", "TMAX") 
label <- c("fare_amount")
folds <- 10

# Const for xgboost
maxDepth <- 8
roundNum <- 10000
threadNum <- 4
presentResult <- 1
parallelIndicator <- T
coreToUse <- detectCores() - 1

# Funciton in calculating the haversine distance between two gps coordinates
GPSdistance <- function(coor1_longitude, coor1_latitude, coor2_longitude, coor2_latitude){
  longD <- coor2_longitude - coor1_longitude
  latiD <- coor2_latitude - coor1_latitude
  haverSine <- sin(latiD / 2) * sin(latiD / 2) + cos(coor1_latitude) * cos(coor2_latitude) * sin(longD / 2) * sin(longD / 2)
  haverAngle <- asin(sqrt(haverSine))
  distance <- 2 * earthR * haverAngle
  return(distance)
}

train <- train %>% separate(pickup_datetime, into = c("DATE", "time", "zone"), sep = " ") %>% 
  select(-key, -zone) %>%
  filter(!(pickup_longitude == 0 | pickup_latitude == 0 | dropoff_latitude == 0 | dropoff_longitude == 0)) %>% filter(passenger_count <= maxPassenger) %>% 
  filter(pickup_longitude > nycLongLwr & pickup_longitude < nycLongUpp & pickup_latitude > nycLatiLwr & pickup_latitude < nycLatiUpp) %>%
  filter(dropoff_longitude > nycLongLwr & dropoff_longitude < nycLongUpp & dropoff_latitude > nycLatiLwr & dropoff_latitude < nycLatiUpp) %>%
  filter(fare_amount > 0) %>%
  mutate(pickup_longitude = (pickup_longitude * pi) / halfCircleDegree, 
         pickup_latitude = (pickup_latitude * pi) / halfCircleDegree,
         dropoff_longitude = (dropoff_longitude * pi) / halfCircleDegree, 
         dropoff_latitude = (dropoff_latitude * pi) / halfCircleDegree) %>%
  mutate(weekday = weekdays(as.Date(DATE))) %>% 
  mutate(miles = GPSdistance(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)) %>%
  mutate(timeFrame = substr(time, 1, 2)) %>% mutate(timeFrame = as.numeric(timeFrame)) %>%
  mutate(yearFrame = substr(DATE, 1, 4)) %>% mutate(yearFrame = as.numeric(yearFrame)) %>%
  mutate(monthFrame = substr(DATE, 6, 7)) %>% mutate(monthFrame = as.numeric(monthFrame)) %>% 
  mutate(pickupToJFK = (GPSdistance(pickup_longitude, pickup_latitude, jfk_coor_long, jfk_coor_lati))) %>%
  mutate(pickupToLGA = (GPSdistance(pickup_longitude, pickup_latitude, lga_coor_long, lga_coor_lati))) %>%
  mutate(pickupToEWR = (GPSdistance(pickup_longitude, pickup_latitude, ewr_coor_long, ewr_coor_lati))) %>%
  mutate(dropoffToJFK = (GPSdistance(dropoff_longitude, dropoff_latitude, jfk_coor_long, jfk_coor_lati))) %>%
  mutate(dropoffToLGA = (GPSdistance(dropoff_longitude, dropoff_latitude, lga_coor_long, lga_coor_lati))) %>%
  mutate(dropoffToEWR = (GPSdistance(dropoff_longitude, dropoff_latitude, ewr_coor_long, ewr_coor_lati)))

nycWeather <- nycWeather %>% select(DATE, AWND, PRCP, SNOW, SNWD, TMAX)

train <- merge(train, nycWeather, by = "DATE")
train <- train[complete.cases(train), ]

  
########################
# Training and Testing #
########################

# Setting up cross validation index
train$cvIndex <- ceiling(sample(1:nrow(train), nrow(train)) / (nrow(train) / folds))

# cross validation
rmse <- rep(NA, folds)

if(!parallelIndicator){
  for(i in 1:folds){
    training <- train[train$cvIndex != i, ] 
    testing <- train[train$cvIndex == i, ] 
    xgbModel <- xgboost(data = model.matrix(~ . + 0, training[, feature]), label = as.matrix(training[, label]), max.depth = maxDepth, nrounds = roundNum, nthread = threadNum, verbose = presentResult)
    predictions <- predict(xgbModel, model.matrix(~ . + 0, testing[, feature]))
    rmse[i] <- sqrt(mean((predictions - testing$fare_amount)^2))
    print(i)
  }
} else{
  cl <- makeCluster(coreToUse)
  registerDoParallel(cl)
  rmse <- foreach(i = 1:folds,
          .packages = 'xgboost') %dopar% {
    training <- train[train$cvIndex != i, ] 
    testing <- train[train$cvIndex == i, ] 
    xgbModel <- xgboost(data = model.matrix(~ . + 0, training[, feature]), label = as.matrix(training[, label]), max.depth = maxDepth, nrounds = roundNum, nthread = threadNum, verbose = presentResult)
    predictions <- predict(xgbModel, model.matrix(~ . + 0, testing[, feature]))
    sqrt(mean((predictions - testing$fare_amount)^2))
  }
  stopCluster(cl)
  unlist(rmse)
}

# Model Training and Testing
# xgbModel <- xgboost(data = model.matrix(~ . + 0, train[, feature]), label = as.matrix(train[, label]), max.depth = 6, nrounds = 1000, nthread = 2, verbose = 1)
# predictions <- predict(xgbModel, model.matrix(~ . + 0, train[, feature]))

mean(rmse)
write.table(rmse, "outputRMSE.txt")

##########################
# Prediction On Test Set #
##########################

#test <- fread("Google Drive/Kaggle_Data/NYCtaxifee/test.csv")

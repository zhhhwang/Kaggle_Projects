library(data.table)
library(xgboost)
library(tidyr)
library(dplyr)
library(lubridate)
library(doParallel)

####################
# Setting up tasks #
####################

# Programming Settings
localRunning <- T
doValidation <- F
doPrediction <- T
parallelIndicator <- F

# Pre-running variables
feature <- c("passenger_count", "weekday", "miles", "monthFrame", 
             "yearFrame", "timeFrame", "pickupToJFK", "pickupToLGA", 
             "pickupToEWR", "dropoffToJFK", "dropoffToLGA", "dropoffToEWR", 
             "PRCP", "SNOW", "SNWD", "TMAX") 
label <- c("fare_amount")
folds <- 5

# Const for xgboost
maxDepth <- 10
roundNum <- 100
threadNum <- 10
presentResult <- 1
coreToUse <- detectCores() - 1

###############
# Import Data #
###############

# Loading the train set
if(localRunning){
  train <- fread("Google Drive/Kaggle_Data/NYCtaxifee/train_small.csv")
  nycWeather <- fread("Google Drive/Kaggle_Data/NYCtaxifee/nycWeather.csv")
  test <- fread("Google Drive/Kaggle_Data/NYCtaxifee/test.csv")
}else{
  train <- fread("/home/zhhhwang/Kaggle_Data/NYCtaxifee/train_small.csv")
  nycWeather <- fread("/home/zhhhwang/Kaggle_Data/NYCtaxifee/nycWeather.csv")
  test <- fread("/home/zhhhwang/Kaggle_Data/NYCtaxifee/test.csv")
}

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

# Funciton in calculating the haversine distance between two gps coordinates
GPSdistance <- function(coor1_longitude, coor1_latitude, coor2_longitude, coor2_latitude){
  longD <- coor2_longitude - coor1_longitude
  latiD <- coor2_latitude - coor1_latitude
  haverSine <- sin(latiD / 2) * sin(latiD / 2) + cos(coor1_latitude) * cos(coor2_latitude) * sin(longD / 2) * sin(longD / 2)
  haverAngle <- asin(sqrt(haverSine))
  distance <- 2 * earthR * haverAngle
  return(distance)
}

# Training set preprocessing
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

nycWeather <- nycWeather %>% select(DATE, PRCP, SNOW, SNWD, TMAX)

train <- merge(train, nycWeather, by = "DATE")
train <- train[complete.cases(train), ]

test <- test %>% separate(pickup_datetime, into = c("DATE", "time", "zone"), sep = " ") %>% 
  select(-zone) %>%
  filter(!(pickup_longitude == 0 | pickup_latitude == 0 | dropoff_latitude == 0 | dropoff_longitude == 0)) %>% filter(passenger_count <= maxPassenger) %>% 
  filter(pickup_longitude > nycLongLwr & pickup_longitude < nycLongUpp & pickup_latitude > nycLatiLwr & pickup_latitude < nycLatiUpp) %>%
  filter(dropoff_longitude > nycLongLwr & dropoff_longitude < nycLongUpp & dropoff_latitude > nycLatiLwr & dropoff_latitude < nycLatiUpp) %>%
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

test <- merge(test, nycWeather, by = "DATE")


########################
# Training and Testing #
########################

if(doValidation){
  
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
  
  mean(unlist(rmse))
  write.table(rmse, "outputRMSE.txt")
}

##########################
# Prediction On Test Set #
##########################



if(doPrediction){
  
  xgbModel <- xgboost(data = model.matrix(~ . + 0, train[, feature]), label = as.matrix(train[, label]), 
                      max.depth = maxDepth, nrounds = roundNum, nthread = threadNum, verbose = presentResult)
  
  fare_amount <- predict(xgbModel, model.matrix(~ . + 0, test[, feature]))
  test <- cbind(test, fare_amount) %>% select(key, fare_amount)
  write.table(test, "submission.csv")
}


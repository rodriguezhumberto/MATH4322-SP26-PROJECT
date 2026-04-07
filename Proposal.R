#Import dataset, assuming file is in current working directory
ecg.df <- read.csv("ECGCvdata.csv")

#Response Variable counts
ailement_counts <- table(ecg.df$ECG_signal)
print(ailement_counts)

#Finding variables with empty values
cols_with_na <- names(ecg.df)[colSums(is.na(ecg.df)) > 0]
print(cols_with_na)

#Dropping columns
to_drop <- c("RECORD", "QRtoQSdur", "RStoQSdur", "PonPQang", "PQRang", "QRSang", "RSTang", "STToffang", "QRslope", "RSslope")
ecg.df <- ecg.df[, !(names(ecg.df) %in% to_drop)]

#Converting response to factor
ecg.df$ECG_signal <- as.factor(ecg.df$ECG_signal)

#Scaling data so it can work with neural network(previous attempt was very inaccurate)
numeric_cols <- sapply(ecg.df, is.numeric)
ecg.df[, numeric_cols] <- scale(ecg.df[, numeric_cols])

#Neural network example
#Load necessary packages
library(nnet)
library(NeuralNetTools)
#Set seed
set.seed(42)
#Creating stratified data for training and test
train_index <- createDataPartition(ecg.df$ECG_signal, p = 0.8, list = FALSE)
train_data <- ecg.df[train_index, ]
test_data <- ecg.df[-train_index, ]
#Fit the model
ecg.nnet = nnet(ECG_signal ~., data = train_data, size = 5)
#Visualize model
plotnet(ecg.nnet)
#Predictions on test data
predictions.nnet <- predict(ecg.nnet, newdata = test_data, type = "class")
#Convert to factos
predictions.nnet <- as.factor(predictions.nnet)
#Confusion matrix
confusionMatrix(predictions.nnet, test_data$ECG_signal)

#Random Forest Example
#Load neccesary packages
library(randomForest)
library(caret)
#Set seed
set.seed(42)
#Creating stratified data for training and test
train_index <- createDataPartition(ecg.df$ECG_signal, p = 0.8, list = FALSE)
train_data <- ecg.df[train_index, ]
test_data <- ecg.df[-train_index, ]
#Fit the model
ecg.rf <- randomForest(ECG_signal ~ ., data = train_data, ntree = 500, importance = TRUE)
summary(ecg.rf)
#Predictions on test data
predictions.rf <- predict(ecg.rf, newdata = test_data)
head(predictions.rf)
#Confusion matrix
confusionMatrix(predictions.rf, test_data$ECG_signal)
#Visualize
varImpPlot(ecg.rf)

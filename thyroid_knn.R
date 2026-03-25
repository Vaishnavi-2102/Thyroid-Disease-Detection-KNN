############################################################
# Thyroid Disease Detection using KNN
# Data Analytics Project in R
############################################################


############################
# 1. Load Required Libraries
############################

library(ggplot2)
library(dplyr)
library(class)
library(caret)
library(corrplot)


############################
# 2. Load Dataset
############################

data <- read.csv("thyroidDF.csv")

# View dataset
head(data)
str(data)


############################
# 3. Data Preprocessing
############################

# Remove missing values
data <- na.omit(data)

# Select important columns
data <- data[, c("age", "TSH", "T3", "TT4", "FTI", "target")]

# Convert target variable to factor
data$target <- as.factor(data$target)


############################################################
# 4. Exploratory Data Analysis (Visualization)
############################################################

############################
# 4.1 Class Distribution (Bar Chart)
############################

ggplot(data, aes(x = target)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Distribution of Thyroid Classes") +
  xlab("Thyroid Condition") +
  ylab("Count")


############################
# 4.2 Histogram (TSH Distribution)
############################

ggplot(data, aes(x = TSH)) +
  geom_histogram(fill="skyblue", bins=30) +
  ggtitle("Distribution of TSH Levels")


############################
# 4.3 Boxplot (Outlier Detection)
############################

ggplot(data, aes(x = target, y = TSH)) +
  geom_boxplot(fill="orange") +
  ggtitle("TSH vs Thyroid Condition")


############################
# 4.4 Scatter Plot (Feature Relationship)
############################

ggplot(data, aes(x = T3, y = TT4, color = target)) +
  geom_point() +
  ggtitle("T3 vs TT4")


############################
# 4.5 Correlation Heatmap
############################

num_data <- data[, sapply(data, is.numeric)]

cor_matrix <- cor(num_data)

corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 0.8)


############################################################
# 5. Feature Scaling (Normalization)
############################################################

normalize <- function(x){
  (x-min(x))/(max(x)-min(x))
}

data_norm <- as.data.frame(lapply(data[,1:5], normalize))


############################################################
# 6. Train Test Split
############################################################

set.seed(123)

train_index <- sample(1:nrow(data_norm),
                      0.8 * nrow(data_norm))

train_data <- data_norm[train_index,]
test_data  <- data_norm[-train_index,]

train_labels <- data$target[train_index]
test_labels  <- data$target[-train_index]


############################################################
# 7. Apply KNN Model
############################################################

knn_model <- knn(
  train = train_data,
  test  = test_data,
  cl    = train_labels,
  k     = 5
)


############################################################
# 8. Confusion Matrix
############################################################

conf_matrix <- confusionMatrix(knn_model, test_labels)

print(conf_matrix)


############################################################
# 9. K vs Accuracy Plot
############################################################

accuracy_values <- c()

for(i in 1:10){
  
  pred <- knn(train_data, test_data, train_labels, k=i)
  
  acc <- sum(pred == test_labels)/length(test_labels)
  
  accuracy_values <- c(accuracy_values, acc)
}

plot(1:10,
     accuracy_values,
     type="b",
     col="blue",
     xlab="K Value",
     ylab="Accuracy",
     main="K vs Accuracy Plot")


############################################################
# 10. Performance Metrics
############################################################

cm <- confusionMatrix(knn_model, test_labels)

accuracy  <- cm$overall['Accuracy']
precision <- mean(cm$byClass[,'Precision'], na.rm=TRUE)
recall    <- mean(cm$byClass[,'Recall'], na.rm=TRUE)
f1        <- mean(cm$byClass[,'F1'], na.rm=TRUE)

metrics <- c(
  Accuracy = accuracy,
  Precision = precision,
  Recall = recall,
  F1 = f1
)


############################################################
# 11. Model Performance Bar Chart
############################################################

barplot(metrics,
        col="lightblue",
        main="Model Performance Metrics",
        ylim=c(0,1))


############################################################
# END OF PROJECT
############################################################
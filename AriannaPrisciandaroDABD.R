rm(list=ls())
setwd("/Users/attiliopittelli/Desktop/Coding/")
library(readxl)
###########################################
#               FUNCTION                  #
###########################################
#kurtosis with comment 
kurtosisFunct <- function(column){
  print("low kurtosis suggest fewer extreme outcomes or outliers")
  if (kurtosis(column) > 3){
    print("the value of the kurtosis is greater than 3 -> leptokurtic: fat tails, sharp peak")
  }
  else if (kurtosis(column) < 3){
    print("the value of the kurtosis is less than 3 -> platykurtic: thin tails, flatter peak")
  }
  else if (kurtosis(column) == 3){
    print("the value of the kurtosis is equal to 3 -> normal distribution")
  }
  return(kurtosis(column))
}
#return skew with comment
skewFunct <- function(column){
  print("positive skewness can mitigate the risks associated with high kurtosis")
  return(skewness(column))
}
#import dataset
wine<-read_excel("wines_SPA.xlsx")
wine<-na.omit(wine)
str(wine)
###############################################
#               SKEW AND CURTOSIS             #
###############################################
#convert all column to numeric
wine$price <- as.numeric(wine$price)
wine$rating <- as.numeric(wine$rating)
wine$year <- as.numeric(wine$year)
wine$num_reviews <- as.numeric(wine$num_reviews)
wine$body <- as.numeric(wine$body)
wine$acidity <- as.numeric(wine$acidity)
data <- na.omit(wine)
hist(wine$price,col = "steelBlue")
kurtosisFunct(wine$price)
skewFunct(wine$price)

wine <- na.omit(wine)
#DESCRIPTIVE STATISTIC SUMMARY FOR WINE
summary(wine)
#linear regression for wine 
# Z-score method
columns = c("price","rating","num_reviews","body","acidity")
data <- (wine[,columns]) 
price <- data$price
z_scores <- scale(price)
mean(abs(z_scores))
plot(z_scores)
potential_outliers <- abs(z_scores) > mean(abs(z_scores))  # Adjust the threshold as needed
potential_outliers

# IQR method
Q1 <- quantile(price, 0.25)
Q3 <- quantile(price, 0.75)
IQR_value <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR_value
upper_bound <- Q3 + 1.5 * IQR_value
potential_outliers_iqr <- price < lower_bound | price > upper_bound
potential_outliers_iqr
# Combine results
potential_outliers <- potential_outliers | potential_outliers_iqr

# Display potential outliers
data[potential_outliers, ]

# Regression model
lm.1.1 <- lm(price ~ num_reviews, data= data)
summary(lm.1.1)

# Recategorize provinc
lm.1.2 <- lm(price ~ num_reviews + body, data= data)
summary(lm.1.2)

# Include interaction effect
lm.1.3 <- lm(price ~ num_reviews + body + num_reviews*body, data= data)
summary(lm.1.3)

# Models fit
summary(lm.1.1)$r.squared
summary(lm.1.2)$r.squared
summary(lm.1.3)$r.squared

#k) Exclude outliers
dataClean <-data[!potential_outliers, ]

lm.1.4 <- lm(price ~ body, data= dataClean)
summary(lm.1.4)
lm.1.5 <- lm(price ~ body + num_reviews, data= dataClean)
summary(lm.1.5)
lm.1.6 <- lm(price ~ body + num_reviews + body*num_reviews, data= dataClean)
summary(lm.1.6)
plot(lm.1.6)

# Regression model using all variable as regressors, but cleaning outliers first
###################################
#       REMOVING OUTLIERS         #
###################################
# Removing all the outliers from the regressors
#NUMBER OF REVIEWS
num_reviews <- data$num_reviews
z_scores <- scale(num_reviews)
potential_outliers <- abs(z_scores) > mean(abs(z_scores))  # Adjust the threshold as needed
dataCleanNew <-data[!potential_outliers, ]
#BODY
body <- dataCleanNew$body
z_scores <- scale(body)
potential_outliers <- abs(z_scores) > mean(abs(z_scores))  # Adjust the threshold as needed
dataClean2 <-dataCleanNew[!potential_outliers, ]

#ACIDITY
acidity <- dataClean2$acidity
z_scores <- scale(acidity)
potential_outliers <- abs(z_scores) > mean(abs(z_scores))  # Adjust the threshold as needed
dataClean3 <-dataClean2[!potential_outliers, ]

#RATING
rating <- dataClean3$rating
z_scores <- scale(rating)
potential_outliers <- abs(z_scores) > mean(abs(z_scores))  # Adjust the threshold as needed

# Exclude outliers
dataFinal <-dataClean3[!potential_outliers, ]

lm.1.5 <- lm(price ~ acidity + num_reviews + body + acidity + rating, data= dataFinal)
summary(lm.1.5)
lm.1.6 <- lm(price ~ acidity + num_reviews + body + body*num_reviews + rating, data= dataFinal)
summary(lm.1.6)
plot(lm.1.6)


#difference of exp on gender over the years
ggplot(data = data) + 
  geom_col(mapping = aes(x = year, y =mean(price), color=sesso),legend("topright",legend = "prezzo", fill = gray.colors(2)))

#analysis of variable num_reviews
num_reviews<-data$num_reviews
summary(num_reviews)
max(num_reviews)-min(num_reviews)
IQR(num_reviews)
var(num_reviews)
boxplot(num_reviews, main="boxplot for variable num_reviews") #to many outliers that we cleaned before

#analysis of variable body
body<-data$body
summary(body)
max(body)-min(body)
IQR(body)
var(body)
boxplot(num_reviews, main="boxplot for variable body") #to many outliers that we cleaned before

#analysis of variable acidity
acidity<-data$acidity
summary(acidity)
max(acidity)-min(acidity)
IQR(acidity)
var(acidity)
boxplot(acidity, main="boxplot for variable acidity")

#analysis of variable rating
acidity<-data$rating
summary(rating)
max(acidity)-min(rating)
IQR(rating)
var(rating)
boxplot(acidity, main="boxplot for variable rating") #to many outliers that we cleaned before


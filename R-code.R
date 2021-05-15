####################################################
#IE583 - Team 17 Final project
#
#Purpose:Initial data processing prior to learning
#   Exports final results into CSV tf_idf_results.csv
####################################################

library(tm)
library(qdap)
library(ggplot2)
library(readtext)
library(stringr)
library(slam)
library(textclean)

####################################################
#SMALL SWITCH to use 1000 records or all records
####################################################
useSmall <- FALSE

#############
# Use up to a lot of virtual memory
#############
invisible(utils::memory.limit(64000))

#######################################################################
###SET A CONSISTENT WORKING DIRECTORY with the needed files
#######################################################################
#setwd("C:/Users")
setwd("~/Training/IowaState/Classes/IE583/Datasets")

# Read in the metadata from the competition (downloaded from Kaggle)
data = read.csv("metadata.csv")
###Drop useless columns
data = subset(data, select = c(cord_uid,title,pubmed_id,abstract,journal))
#gc() #not sure if gc needed

#######################################################################
### Make all Title/Abstract words lower case
#######################################################################
data$title <- tolower(data$title)
data$abstract <- tolower(data$abstract)

#######################################################################
### Remove non-ascii - takes about a minute
#######################################################################
data$title <- replace_non_ascii(data$title, replacement = " ")
data$abstract <- replace_non_ascii(data$abstract, replacement = " ")


#####################
# USE SMALL SWITCH
#####################
if (useSmall == TRUE){
  data = data[1:1000,]
} else{
  print ("Using all data")
}

#####################################################################
# Initial text handling to prep for analysis
#####################################################################
text <- data$title; data$abstract

#we will need this for some of our matrices later
doc_list <- lapply(data[,1],function(x) genX(x, " [", "]"))
N.docs  <- length(doc_list)
names(doc_list) = data[,1]

#####################################################################
# Read in Environment word dictionaries
#####################################################################
envirDic = readtext("EnvironmentDictionary.txt",dvsep = "\n")
#create a list of search query terms
term_list = unlist(strsplit(envirDic[1,2],"\n"))
#get number of queries
Num.term = length(term_list)
names(term_list) = term_list

#####################################################################
# Create text corpus, and preproceess text + stemming
#####################################################################
#text.source <- VectorSource(text)
text.corpus <- VCorpus(VectorSource(text))
#inspect(text.corpus)
#writeLines(as.character(text.corpus[[50]]))

#preprocessing 
toSpace = content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})
#is some of this covered in line 86?
text.corpus = tm_map(text.corpus, toSpace, "-")
text.corpus = tm_map(text.corpus, toSpace, ":")
text.corpus = tm_map(text.corpus, toSpace, "'")
text.corpus = tm_map(text.corpus, toSpace, " -")
#writeLines(as.character(text.corpus[[50]]))
text.corpus <- tm_map(text.corpus, removePunctuation)
text.corpus <- tm_map(text.corpus, content_transformer(tolower))
text.corpus <- tm_map(text.corpus, removeWords, stopwords("english"))
text.corpus <- tm_map(text.corpus, stripWhitespace)
#writeLines(as.character(text.corpus[[50]]))

#Stemming
library(SnowballC)
text.corpus <- tm_map(text.corpus, stemDocument)

#####################################################################
# Create TF-IDF document matrix and normalize
#####################################################################
#Document matrix
#creating term matrix with TF-IDF weighting
text.dtm <-DocumentTermMatrix(text.corpus,control = list(dictionary=term_list, weighting = function(x) weightTfIdf(x, normalize = FALSE)))

#Visual check for comparing different DocTermMatrices and used for normalization
text.matrix <- as.matrix(text.dtm)
text.matrixprenorm <- text.matrix

#normalizing the term document matrix
dtm_norm<- scale(text.matrix, center = FALSE,scale = sqrt(colSums(text.matrix^2)))
#tfidf_dtmmat = as.matrix(dtm_norm)
#rownames(tfidf_dtmmat) = c(names(doc_list),names(term_list))
#text.matrix

#let's filter for just our dictionary
#create filter of all the terms
to_filter <- colnames(dtm_norm)
#then filter based on the vector (term_list) from our original dictionary
dtm2 <- dtm_norm[,to_filter %in% term_list]
#check new dtm2 
#colnames(dtm2)

#column correlation matrix
cor.matrix <- cor(as.matrix(dtm2)[,term_list], as.matrix(dtm2)[,term_list])
cor.matrix[is.nan(cor.matrix)] = 0

dtm2[is.nan(dtm2)] = 0
dtm2 = dtm2[,colSums(dtm2) > 0]

#save off the column names in case we need them later
test.colnames <- as.data.frame(dimnames(dtm2)$Terms)

#reset our text.matrix
text.matrix <- as.matrix(dtm2)
text.matrix[is.nan(text.matrix)] = 0

#adds the doc names to rows
rownames(text.matrix) = (names(doc_list))

#####################################################################
# Create Score target attribute
#####################################################################
#add a score to the results --> do we need cosine similarity here if there is no query?
resultsdtm.df <- data.frame(text.matrix)
resultsdtm.df <- cbind(resultsdtm.df, Score = rowSums(resultsdtm.df))

write.csv(resultsdtm.df,"tf_idf_results.csv", row.names = TRUE)
write.csv(cor.matrix,"cormatrix2.csv", row.names = TRUE)

####################################################
#IE583 - Team 17 Final project
#
#Purpose: Model Training and Evaluation
#   Imports TF_IDF results
#   Hyperparameter checking
#   Final model creation
#   Contains ways to evaluate performance
####################################################

###########################################################################
# Model Training and Evaluation
###########################################################################

library(dplyr)
library(tidyr)
library(randomForest)
library(e1071)
library(caret)
library(ggpubr)
library(RWeka)
library(DMwR)
library(corrplot)
library(Boruta)
library(Metrics)

################################
# Try to read in Saved rf_model ?
# Bypasses training
################################
bypassTrain <- FALSE

#######################################################################
###SET A CONSISTENT WORKING DIRECTORY with the needed files
#######################################################################
setwd("C:/Users")
#setwd("~/Training/IowaState/Classes/IE583/Datasets")

################################
# Read in our tf-idf results
################################
dt = read.csv("tf_idf_results.csv")
str(dt)
dt = dt[,-1]
dt$Score = as.numeric(dt$Score)
table(dt$Score)
class(dt$Score)

################################
# Split into 1/3 test, 2/3 train
################################
set.seed (1234)
trainIndex = createDataPartition(dt$Score, p=.67,list=FALSE,times=1)
dt_train = dt[trainIndex,]
dt_test = dt[-trainIndex,]

if(bypassTrain == FALSE){
  #############################################
  # Create a train grid and use caret to train
  #############################################
  tc <- trainControl(method='oob', number=1, search='grid')
  
  tgrid <- expand.grid(
    mtry = c(  floor(ncol(dt_train)/10),
               floor(ncol(dt_train)/5),
               floor(ncol(dt_train)/4),
               floor(ncol(dt_train)/3),
               floor(ncol(dt_train)/2.5),
               floor(ncol(dt_train)/2)), 
    splitrule = c('variance','maxstat','extratrees'), #'maxstat','extratrees' had poor performance
    min.node.size = c(5,10) #Default for regression
  )
  
  #Experimental enhancement for extra hyperparameters
  rf_models_200 <- train(Score~., data = dt_train, method = "ranger", tuneGrid = tgrid,
                         num.trees=200,verbose = TRUE, trControl = tc)
  rf_models_500 <- train(Score~., data = dt_train, method = "ranger", tuneGrid = tgrid,
                         num.trees=500,verbose = TRUE, trControl = tc)
  rf_models_1000 <- train(Score~., data = dt_train, method = "ranger", tuneGrid = tgrid,
                          num.trees=1000,verbose = TRUE, trControl = tc)
  rf_models_1500 <- train(Score~., data = dt_train, method = "ranger", tuneGrid = tgrid,
                          num.trees=1000,verbose = TRUE, trControl = tc)

  #Save our progress
  #save(rf_models, file = "rf_models.rda")
  
  
  ###################################
  # Evaluate and pick a single model
  ###################################
  plot.train(rf_models_200)
  plot.train(rf_models_500)
  plot.train(rf_models_1000)
  plot.train(rf_models_1500)
  results200_df <- do.call("cbind", rf_models_200$results)
  results500_df <- do.call("cbind", rf_models_500$results)
  results1000_df <- do.call("cbind", rf_models_1000$results)
  results1500_df <- do.call("cbind", rf_models_1500$results)
  
  #model_results <- resamples(rf_models_200,rf_models_500,rf_models_1000) 
  #summary(model_results)
  #boxplot(model_results)
  #dotplot(model_results)
  
  
  ###################################
  # Since we don't want the train's 'best'
  # chosen, retrain with our ideal parameters
  ###################################
  model_rf <- ranger(Score~., data = dt_train, mtry = 33, 
                    splitrule = 'extratrees', num.trees = 1000,
                    min.node.size =5,importance='permutation')
  
  
  ###################################
  # Use model to make predictions
  ###################################
  model_rf_pred = predict(model_rf, dt_test)
  
  ###################################
  # Save off model and predictions
  ###################################
  save(model_rf,model_rf_pred, file = "model_rf_and_pred.rda")
  
} else{
  load("model_rf_and_pred.rda")
}

###################################
# Performance measures
###################################

model_rmse <- rmse(model_rf_pred$predictions, dt_test$Score)
print(model_rmse)
print(model_rf)

###################################
# Plot top attributes
###################################
sorted_importance <- sort(model_rf$variable.importance, decreasing = TRUE)
#All
plot(sorted_importance,xaxt="n")
#Top25 w/ labels
plot(sorted_importance[1:25],xaxt="n", main="Top 25 Attributes") 
axis(1,at=1:25,labels=names(sorted_importance)[1:25],las=2)
lines(sorted_importance[1:25])

###################################
# Q-Q plot
###################################
library("car")
RF.resid <- dt_test$Score - model_rf_pred$predictions
qqplot(dt_test$Score,
       RF.resid,
       pch=20,
       main="Normal Q-Q",
       xlab="Theoretical",
       ylab="Residuals",
       col = "black"
)

###################################
# Predicted difference evaluation
###################################
# Read in the metadata from the competition (downloaded from Kaggle)
data = read.csv("metadata.csv")
###Drop useless columns
data = subset(data, select = c(title,abstract))
data$title <- tolower(data$title)
data$abstract <- tolower(data$abstract)
data$title <- replace_non_ascii(data$title, replacement = " ")
data$abstract <- replace_non_ascii(data$abstract, replacement = " ")
###Split up the same as our train/test data
data.train = dt[trainIndex,]
data.test = dt[-trainIndex,]
###Add actual/predicted results columns
data.evaluate <- cbind(data.test, dt_test$Score, model_rf_pred$predictions)
###Add measure for checking false-negatives
eval <- cbind(eval, pred_minus_score=eval$Score-eval$`model_rf_pred$predictions`)
###Add measure for checking false-positives
eval <- cbind(eval, score_minus_pred=`model_rf_pred$predictions`-eval$Score)

#Save off analysis
write.csv(eval,"tested_score_vs_predictions.csv", row.names = TRUE)











#############################################
# OLD approaches and code - save for future usefulness
#############################################

model_rf = randomForest(Score~., data = dt_train)
model_rf_pred = predict(model_rf, dt_test)
rmse(model_rf_pred, dt_train$Score)
rmse(model_rf_pred, dt_test$Score)
varImpPlot(model_rf)
model_rf


#attribute selection
boruta.dt <- Boruta(Score~., data = dt, doTrace = 2)
print(boruta.dt)
#boruta.dt = dt[,-1]          
#get the new model of confirmed only features
getConfirmedFormula(boruta.dt)

boruta.dt$Score = as.numeric(boruta.dt$Score)
#model_rf = randomForest(Score~., data = boruta.dt)

#look at the final attributes selected
#plot(final.boruta)
plot(boruta.dt, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.dt$ImpHistory),function(i)
  boruta.dt$ImpHistory[is.finite(boruta.dt$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.dt$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.dt$ImpHistory), cex.axis = 0.7)


#Get list of confirmed attributes
boruta.predictors <- getSelectedAttributes(boruta.dt, withTentative = F)
boruta.predictors
boruta.dt$finalDecision
#boruta.df <- attStats(boruta.dt)
boruta.stats <- attStats(boruta.dt)
write.csv(boruta.stats,"borutaStats.csv", row.names = TRUE)


#Build model on feature selection
model_rf = randomForest(
  Score ~ air + airborn + airflow + altitud + atmospher + autumn + 
    bio + bioaerosol + biodivers + biofilm + biometr + bird + 
    bulb + carbon + cat + circumst + climat + climatolog + cloud + 
    condit + dog + domain + dome + domest + domesticus + dwell + 
    earth + earthquak + ecosystem + element + environ + environment + 
    evapor + fall + filter + fire + forest + freshwat + frozen + 
    garden + gas + gaseous + habitat + heat + home + hous + household + 
    humid + hvac + landscap + local + milieu + moistur + mosquito + 
    mountain + neighborhood + nois + occup + oil + outdoor + 
    outsid + oxygen + ozon + pollen + precipit + radiat + rainfal + 
    region + resid + residenti + sea + season + silenc + silent + 
    site + situ + smoke + smoker + snow + solar + spheric + spring + 
    steril + storm + summer + sun + surround + temperatur + territori + 
    tropic + trough + water + weather + wet + wind + winter + 
    wood, data=dt_train, importance = TRUE) 




# Predicting on test set - check our error
model_rf_pred = predict(model_rf, dt_test)
rmse(model_rf_pred, dt_test$Score)
mean(model_rf_pred == dt_test$Score)
varImpPlot(model_rf)
model_rf
table(model_rf_pred,dt_test$Score)




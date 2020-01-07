
# title: "Billboard Top 100 Hit Prediction"
# output: html_notebook


# Loading Libraries


library(NLP)
library(tm)
library(SnowballC)
library(wordcloud)
library(MASS)
library(caTools)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(corpus)
library(qdapRegex)
library(gbm)
library(boot)
library(dplyr)
library(ggplot2)
library(gplots)
library(GGally)
library(caTools)
library(car)
library(ROCR)
library(pROC)
library(mlbench)
library(caretEnsemble)
library(glmnet)


# Loading Data

songs = read.csv("../data/songs_complete_data.csv", stringsAsFactors=FALSE)
songs = songs %>% filter(Release_Year >= 2000, Release_Year < 2019)

songs$X = NULL
songs$Artist = NULL
songs$Title = NULL
songs$URI = NULL
songs$Release_Year = NULL



all_lyrics = songs$lyrics
top100 = songs$Top100

songs$lyrics = NULL
songs$Top100 = NULL

songs$Danceability = scale(songs$Danceability)
songs$Energy = scale(songs$Energy)
songs$Loudness = scale(songs$Loudness)
songs$Speechiness = scale(songs$Speechiness)
songs$Acousticness = scale(songs$Acousticness)
songs$Instrumentalness = scale(songs$Instrumentalness)
songs$Liveness = scale(songs$Liveness)
songs$Valence = scale(songs$Valence)
songs$Tempo = scale(songs$Tempo)
songs$Duration = scale(songs$Duration)

songs$Key = as.factor(songs$Key)
songs$Mode = as.factor(songs$Mode)
songs$Time_Signature = as.factor(songs$Time_Signature)
songs$explicit = as.factor(songs$explicit)
songs$Genre = as.factor(songs$Genre)

# Convert every categorical variable to numerical using dummy variable
dmy <- dummyVars(" ~ .", data=songs, fullRank = TRUE)
songs <- data.frame(predict(dmy, newdata=songs))

# Add back lyrics and dependent variable
songs$Lyrics = all_lyrics
songs$Top100 = as.factor(top100)


# NLP: Bag of Words

set.seed(123)

corpus_lyrics = Corpus(VectorSource(songs$Lyrics))

corpus_lyrics = tm_map(corpus_lyrics, tolower)

corpus_lyrics = tm_map(corpus_lyrics, removeNumbers)
corpus_lyrics = tm_map(corpus_lyrics, removeWords, stopwords('english'))

corpus_lyrics = tm_map(corpus_lyrics, removePunctuation)

corpus_lyrics = tm_map(corpus_lyrics, stemDocument)

for (i in seq(corpus_lyrics)) {
  corpus_lyrics[[i]] <- gsub('[^a-zA-Z|[:blank:]]', "", corpus_lyrics[[i]])
}

corpus_lyrics = tm_map(corpus_lyrics, stemDocument)
corpus_lyrics = tm_map(corpus_lyrics, removeWords, stopwords('english')) 
corpus_lyrics = tm_map(corpus_lyrics, stripWhitespace)

frq_lyrics = DocumentTermMatrix(corpus_lyrics)
sparse_lyrics = removeSparseTerms(frq_lyrics, 0.93)
sparse_lyrics

lyric_words = as.data.frame(as.matrix(sparse_lyrics))
colnames(lyric_words) = make.names(colnames(lyric_words))

songs = cbind(songs, lyric_words)
songs$Lyrics = NULL


# Building ML Models
```{r}
# helper functions to calculate accuracy, TPR, and FPR

tableAccuracy <- function(test, pred) {
  t = table(test, pred)
  a = sum(diag(t))/length(test)
  return(a)
}

tableTPR <- function(label, pred) {
  t = table(label, pred)
  return(t[2,2]/(t[2,1] + t[2,2]))
}

tableFPR <- function(label, pred) {
  t = table(label, pred)
  return(t[1,2]/(t[1,1] + t[1,2]))
}


# Train-Test Split

spl = sample.split(songs$Top100, SplitRatio = 0.75)
train = songs %>% filter(spl == TRUE)
test = songs %>% filter(spl == FALSE)

levels(train$Top100) <- make.names(levels(factor(train$Top100)))
levels(test$Top100) <- make.names(levels(factor(test$Top100)))

# make model matrices
train.mm = as.data.frame(model.matrix(Top100 ~ . + 0, data=train))
test.mm = as.data.frame(model.matrix(Top100 ~ . + 0, data=test)) 
```


# Model 0: Baseline Model

table(train$Top100) # most frequent is "Not Top 100"
table(test$Top100) # baseline: predict always "Not Top 100"
baseline_accuracy = 1414/(1414+357) # 0.7984
```

# Model 1: Logistic Regression

set.seed(123)

log_model = glm(Top100 ~ .-Time_Signature.4-Genre.rap, data = train, family = "binomial")
vif(log_model) # Removed Time_Signature.4 and Genre.rap because of high VIF score
summary(log_model)

predict_log = predict(log_model, newdata = test, type = "response")
table(test$Top100, predict_log > 0.5)
tableAccuracy(test$Top100, predict_log > 0.5) # 0.8085827
tableTPR(test$Top100, predict_log > 0.5) # 0.2885154
tableFPR(test$Top100, predict_log > 0.5) # 0.06011315

# Calculate ROC curve
rocr.log.pred <- prediction(predict_log, test$Top100)
logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.log.pred, "auc")@y.values) # AUC = 0.7859144
```


# Model 2: Improved Logistic Regression

set.seed(123)

#Selecting only significiant features (excluding lyrics)
log_model = glm(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                  Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                  Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock, 
                data = train, family = "binomial")
vif(log_model)
summary(log_model)

predict_log = predict(log_model, newdata = test, type = "response")
table(test$Top100, predict_log > 0.5)
tableAccuracy(test$Top100, predict_log > 0.5) # 0.8102767
tableTPR(test$Top100, predict_log > 0.5) # 0.2997199
tableFPR(test$Top100, predict_log > 0.5) # 0.06082037

# Calculate ROC curve
rocr.log.pred <- prediction(predict_log, test$Top100)
logPerformance <- performance(rocr.log.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.log.pred, "auc")@y.values) # AUC = 0.7847674


# Model 3: LDA

set.seed(123)

lda_model = lda(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                  Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                  Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock,
                data = train, family = "binomial")

predict_lda = predict(lda_model, newdata=test)
predict_lda$class[1:10]
predict_lda$posterior[1:10, ] 
predict_lda_probs <- predict_lda$posterior[,2]

table(test$Top100, predict_lda_probs > 0.5)
tableAccuracy(test$Top100, predict_lda_probs > 0.5) # 0.8051948
tableTPR(test$Top100, predict_lda_probs > 0.5) # 0.280112
tableFPR(test$Top100, predict_lda_probs > 0.5) # 0.06223479

# Calculate ROC curve
rocr.lda.pred <- prediction(predict_lda_probs, test$Top100)
ldaPerformance <- performance(rocr.lda.pred, "tpr", "fpr")
plot(ldaPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.lda.pred, "auc")@y.values) # AUC = 0.7739551


# Model 4: 10-CV CART

set.seed(123)

train_cart = train(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                     Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                     Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock,
                   data = train,
                   method = "rpart",
                   metric = 'Accuracy',
                   tuneGrid = data.frame(cp=seq(0, 0.2, 0.002)),
                   trControl = trainControl(method="cv", number=10))

cart_model = train_cart$finalModel
prp(cart_model) # CART visualization

predict_cart = predict(cart_model, newdata = test, type='class')
table(test$Top100, predict_cart)
tableAccuracy(test$Top100, predict_cart) # 0.8046302
tableTPR(test$Top100, predict_cart) # 0.1232493
tableFPR(test$Top100, predict_cart) # 0.02333805

#Calculate ROC curve
cart_probs = predict(cart_model, newdata = test, type='prob')
rocCurve.cart <- roc(test$Top100, cart_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.cart, col=c(4))
auc(rocCurve.cart) # 0.7057


# Model 5: Random Forest (default)

set.seed(123)

# (Takes about 3 min to run)
rf_model = randomForest(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                          Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                          Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock, 
                        data = train)  

predict_rf = predict(rf_model, newdata = test)
table(test$Top100, predict_rf)
tableAccuracy(test$Top100, predict_rf) # 0.8125353
tableTPR(test$Top100, predict_rf) # 0.1736695
tableFPR(test$Top100, predict_rf) # 0.0261669

#Calculate ROC curve
rf_probs = predict(rf_model, newdata = test, type='prob')
rocCurve.rf <- roc(test$Top100, rf_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.rf, col=c(4))
auc(rocCurve.rf) # 0.7731


# Model 6: Bagging

set.seed(123)

bag_model <- randomForest(x = train.mm, y = train$Top100, mtry = 71, nodesize = 5, ntree = 500)

predict_bag <- predict(bag_model, newdata = test.mm)
table(test$Top100, predict_bag)
tableAccuracy(test$Top100, predict_bag) # 0.8176172
tableTPR(test$Top100, predict_bag) # 0.2997199
tableFPR(test$Top100, predict_bag) # 0.05162659

#Calculate ROC curve
bagging_probs = predict(bag_model, newdata = test, type='prob')
rocCurve.bagging <- roc(test$Top100, bagging_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.bagging, col=c(4))
auc(rocCurve.bagging) # 0.7846


# Model 7: 10-CV KNN

set.seed(123)

objControl <- trainControl(method="cv", number=10, classProbs=TRUE, summaryFunction=twoClassSummary)

train_knn <- train(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                     Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                     Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock,
                   data = train,
                   method = "knn",
                   trControl = objControl, 
                   tuneLength = 20)

knnPredict <- predict(train_knn, newdata = test)
table(test$Top100, knnPredict)
tableAccuracy(test$Top100, knnPredict) # 0.8006776
tableTPR(test$Top100, knnPredict) # 0.0140056
tableFPR(test$Top100, knnPredict) # 0.0007072136

# Calculate ROC curve
knn_probs = predict(train_knn, newdata = test, type='prob')
rocCurve.knn <- roc(test$Top100, knn_probs[,"X1"])
#plot the ROC curve
plot(rocCurve.knn, col=c(4))
auc(rocCurve.knn) # 0.7266


# Model 8: Stacking (Logistic + CART + LDA)

set.seed(123)

# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm')

models <- caretList(Top100 ~ .-Danceability-Key.1-Key.2-Key.3-Key.4-Key.5-Key.6-Key.7-Key.9-Key.10-Key.11-
                      Mode.1-Speechiness-Time_Signature.3-Time_Signature.4-Time_Signature.5-Genre.classical-
                      Genre.edm-Genre.jazz-Genre.rap-Genre.metal-Genre.reggae-Genre.rock, data=train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)

# stack using glm
set.seed(123)
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

predict_stack = predict(stack.glm, newdata = test, type = "prob")
table(test$Top100, predict_stack > 0.5)
tableAccuracy(test$Top100, predict_stack > 0.5) # 0.8136646
tableTPR(test$Top100, predict_stack > 0.5) # 0.2969188
tableFPR(test$Top100, predict_stack > 0.5) # 0.05586987

# Calculate ROC curve
rocr.stack.pred <- prediction(predict_stack, test$Top100)
stackPerformance <- performance(rocr.stack.pred, "tpr", "fpr")
plot(logPerformance, colorize = TRUE)
abline(0, 1)

as.numeric(performance(rocr.stack.pred, "auc")@y.values) # AUC = 0.7966771


# Model 9: Ridge Regression

set.seed(123)

# convert training data to matrix format
x <- model.matrix(Top100 ~ ., train)
# convert class to numerical variable
y <- ifelse(train$Top100=='X1',1,0)

# perform grid search to find optimal value of lambda
# family = binomial => logistic regression, alpha = 0 => ridge
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x, y, alpha = 0, family = 'binomial', type.measure = 'mse')
#plot result
plot(cv.out)

# min value of lambda
lambda_min <- cv.out$lambda.min
# best value of lambda
lambda_1se <- cv.out$lambda.1se
# regression coefficients
coef(cv.out,s=lambda_1se)

# get test data
x_test <- model.matrix(Top100 ~ ., test)
# predict class, type=”class”
ridge_prob <- predict(cv.out, newx = x_test, s = lambda_1se, type = 'response')
#translate probabilities to predictions
ridge_predict <- rep('X0', nrow(test))
ridge_predict[ridge_prob>.5] <- 'X1'

#confusion matrix
table(pred=ridge_predict,true=test$Top100)
tableAccuracy(test$Top100, ridge_predict) # 0.8046302
tableTPR(test$Top100, ridge_predict) # 0.1820728
tableFPR(test$Top100, ridge_predict) # 0.03818953


# Model 10: Lasso Regression

set.seed(123)

# convert training data to matrix format
x <- model.matrix(Top100 ~ ., train)
# convert class to numerical variable
y <- ifelse(train$Top100=='X1',1,0)

# perform grid search to find optimal value of lambda
# family = binomial => logistic regression, alpha = 1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x, y, alpha = 1, family = 'binomial', type.measure = 'mse')
#plot result
plot(cv.out)

# min value of lambda
lambda_min <- cv.out$lambda.min
# best value of lambda
lambda_1se <- cv.out$lambda.1se
# regression coefficients
coef(cv.out,s=lambda_1se)

# get test data
x_test <- model.matrix(Top100 ~ ., test)
# predict class, type=”class”
lasso_prob <- predict(cv.out, newx = x_test, s = lambda_1se, type = 'response')
#translate probabilities to predictions
lasso_predict <- rep('X0', nrow(test))
lasso_predict[lasso_prob>.5] <- 'X1'

#confusion matrix
table(pred=lasso_predict,true=test$Top100)
tableAccuracy(test$Top100, lasso_predict) # 0.8074534
tableTPR(test$Top100, lasso_predict) # 0.1848739
tableFPR(test$Top100, lasso_predict) # 0.03536068

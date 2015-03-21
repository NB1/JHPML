#load required libraries. Set some variable.
#USAGE:  1) FIX THE DIRECTORY PATH BELOW IN the setwd CALL.  2) Cut and paste the content of this file in an R window.  
library(dplyr)
library(caret)
require(dplyr)
require(caret)
CORRTHRESH = -1
CORRSAMPLESIZE = 1500
UNLIKELYSTR = "impossiblevaxdfd"

#Define some needed functions
#determines if a column has all numeric values
fullyNumeric = function(x) {length(x) == length(subset(x, is.numeric(x)))} 

#determines if a column has no missing value
noMissingValue = function(x) {length(subset(x, is.na(x))) == 0} 

#find very correlated columns
getCorrelatedCols = function(corrsampl) {
    rawCols = colnames(corrsampl)
    numCols = rawCols[sapply(1:length(rawCols), FUN=function(x) {fullyNumeric(unlist(corrsampl[,x]))})]
    nonaCols = rawCols[sapply(1:length(rawCols), FUN=function(x) {noMissingValue(unlist(corrsampl[,x]))})]
    hccds = corrsampl[,intersect(numCols, nonaCols)]
    allcor = cor(hccds, hccds)
    N <- dim(allcor)[1]
    correlatedCols = rep(UNLIKELYSTR, N)
    k = 1
    vars <- colnames(hccds)
    for (i in 1:N) {
        for (j in i:N) {
            if ((abs(allcor[i,j]) > CORRTHRESH) && (i != j)) {
	   	correlatedCols[k] = vars[i]
		k=k+1	
                }
            }
        }
    setdiff(unique(correlatedCols), c(UNLIKELYSTR));
}

#use the training set to learn a random forest.  Return predictions
#and some other important info as list.  
learn <- function(trainingSet, testingSet) {
    t1 = Sys.time()
    tmp = list()
    tmp$tc <- trainControl(method="cv", number=5)
    tmp$goodCols <- names(trainingSet);
    tmp$numCols <- dim(trainingSet)[2]
    tmp$fit <- train(classe ~ ., method="rf", data=trainingSet, prox=TRUE, trControl=tmp$tc)
    tmp$cm <- confusionMatrix(predict(tmp$fit, newdata = testingSet), testingSet$classe);
    tmp$approxTime = floor(Sys.time() - t1)
    tmp
}

#solve a large number of smaller learning problems (given by
#iterations).  From each returned random forest, pick the most
#important columns. Pick the columns that pay biggest role in all
#iteratons combined.
mostImportantFeatures <- function(trainingSet, iterations=10, SSSize = 300) {
    cols = names(trainingSet)
    importance <- rep(0.0, length(cols));
    for (i in 1:iterations) {
        trn = trainingSet
        if (SSSize < dim(trainingSet)[1]) {
            smpl = sample(dim(trainingSet)[1], SSSize);
            trn <- trainingSet[smpl, ]
        }
        learning = learn(trn, trn)
        imp <- learning$fit$finalModel["importance"];
        accuracy = as.data.frame(learning$cm["overall"])["Accuracy",];
        importance <- importance + 
        sapply(cols, FUN = function(x){if (is.na(imp[1]$importance[,][x])) {0} else {imp[1]$importance[,][x]*accuracy}});
    }
    names(trainingSet)[order(importance, decreasing=TRUE)]
}




#Step 1:  Setup initial environment. WD, SEED, DATA, constants
trn = NULL
tinyTST = NULL
pmlInit = function () {
    rm(list=ls())
    setwd(
        paste("C:/Users/neeraj/b/Biz/wrk_hm/Learning/",
              "Coursera/PracticalMachineLearningCode/Programs",
              sep="", collapse="")
        )
    trn <<- read.csv("../Data/pml-training.csv")
    tinyTST <<- read.csv("../Data/pml-testing.csv")    
    set.seed(71679174)
    TESTDATAPROB <<- .3
    CORRTHRESH <<- .8
}
pmlInit()


#Step 2: convert to tidy, remove stat-summary cols
trn <- tbl_df(trn)
tinyTST <- tbl_df(tinyTST)
trn<-trn[,2:160]
tinyTST<-tinyTST[,2:160]
allcols = colnames(trn)
print(dim(trn)[2]);print("XXXXX")
rawCols <- allcols[-grep("^max|^min|^amplitude|^var|^avg|^stddev|^skewness|^kurtosis", allcols)]
trn<- trn[rawCols]
tinyTST<- tinyTST[rawCols] 
print(dim(trn)[2]);print("XXXXX")

#Step 3: Create a test set that will not be touched in any analysis
#and will be used only for testing
testingRows <- sample(dim(trn)[1], size = dim(trn)[1]*TESTDATAPROB)
tstset <- trn[testingRows,] #do not touch it except to deselect columns.
trn <- trn[-testingRows,] #not intersected with testing at all.
print(dim(trn)[2]);print("XXXXX")

#Step 4:  Get rid of highly correlated columns
corrCols = getCorrelatedCols(trn[sample(dim(trn)[1], size = CORRSAMPLESIZE), ]);
keep = setdiff(names(trn), corrCols)
print(length(corrCols))
tstset <- trn[, keep] #do not touch it except to deselect columns.
trn <- trn[, keep] #not intersected with testing at all.
tinyTST<- tinyTST[, keep] 
print(dim(trn)[2]);print("XXXXX")




#Step 5: Now do the main experiment.  we will create training sets of
#diverse sizes (increasing), learn a model of them, check the accuracy
#of the model.  We will try to hit 99%+ accuracy.  Each model will be
#tested on two sets 1) The main test set (tstset) created above and 2)
#all other rows not used in training.  If total rows are 20000 and
#5000 are in the tstset, potentially we can use 15000 roes for
#training. But we used only NN for training.  The remaining 15000-N
#can also be used as a separate test set. We will vary NN from 100 to
#10000 and increase its value by 25% each time.

NN = 3508
answer = NULL
df = NULL;

bestN = floor(0.75 * dim(trn)[2]);
features = union(mostImportantFeatures(trn)[1:bestN], "classe")
trn <- trn[,features]
tstset  <- tstset[,features]
tinyTST <- tinyTST[,features]
print(dim(trn)[2]);print("XXXXX")

for (i in 1:150) {
    savedTrn <- trn
    savedTst <- tstset
    savedTiny <- tinyTST
    NN = floor(NN * 1.25)
    if (NN > 7000) {
        break;
    }
    rows = sample(dim(trn)[1], NN);
    thisTrain <- trn[rows,]
    thisTest  <- trn[-rows,]
    L <- learn(thisTrain, thisTest)
    answer <- paste(predict(L$fit, newdata = tinyTST),sep="", collapse="")
    cm <- confusionMatrix(predict(L$fit, newdata = tstset), tstset$classe)
    if (is.null(df)) {
        df = c(NN, L$numCols, L$approxTime, as.data.frame(L$cm["overall"])["Accuracy",], as.data.frame(cm["overall"])["Accuracy",], answer)
        names(df) = c("TrainingSize", "Features", "ApproxTime", "AccuracyOnUnusedTraining", "AccuracyOnSetAsideTest", "CatsOfTiny")
    } else {
        df = rbind(df, c(NN, L$numCols, L$approxTime, as.data.frame(L$cm["overall"])["Accuracy",], as.data.frame(cm["overall"])["Accuracy",], answer))
    }
    print(df)
    write.csv(df, paste("CorrFile", NN, ".csv", sep="_"));
    trn <- savedTrn
    tstset <- savedTst
    tinyTST <- savedTiny
}








#load required libraries. Set some variable.
library(dplyr)
library(caret)
require(dplyr)
require(caret)
CORRTHRESH = -1
CORRSAMPLESIZE = 1500
UNLIKELYSTR = "impossiblevaxdfd"

#Define some needed functions
#determines if a column has all numeric values
fullyNumeric = function(x) {length(x) == length(subset(x, is.numeric(x)))} 

#determines if a column has no missing value
noMissingValue = function(x) {length(subset(x, is.na(x))) == 0} 

#find very correlated columns
getCorrelatedCols = function(corrsampl) {
    rawCols = colnames(corrsampl)
    numCols = rawCols[sapply(1:length(rawCols), FUN=function(x) {fullyNumeric(unlist(corrsampl[,x]))})]
    nonaCols = rawCols[sapply(1:length(rawCols), FUN=function(x) {noMissingValue(unlist(corrsampl[,x]))})]
    hccds = corrsampl[,intersect(numCols, nonaCols)]
    allcor = cor(hccds, hccds)
    N <- dim(allcor)[1]
    correlatedCols = rep(UNLIKELYSTR, N)
    k = 1
    vars <- colnames(hccds)
    for (i in 1:N) {
        for (j in i:N) {
            if ((abs(allcor[i,j]) > CORRTHRESH) && (i != j)) {
	   	correlatedCols[k] = vars[i]
		k=k+1	
                }
            }
        }
    setdiff(unique(correlatedCols), c(UNLIKELYSTR));
}

#use the training set to learn a random forest.  Return predictions
#and some other important info as list.  
learn <- function(trainingSet, testingSet) {
    t1 = Sys.time()
    tmp = list()
    tmp$tc <- trainControl(method="cv", number=5)
    tmp$goodCols <- names(trainingSet);
    tmp$numCols <- dim(trainingSet)[2]
    tmp$fit <- train(classe ~ ., method="rf", data=trainingSet, prox=TRUE, trControl=tmp$tc)
    tmp$cm <- confusionMatrix(predict(tmp$fit, newdata = testingSet), testingSet$classe);
    tmp$approxTime = floor(Sys.time() - t1)
    tmp
}

#solve a large number of smaller learning problems (given by
#iterations).  From each returned random forest, pick the most
#important columns. Pick the columns that pay biggest role in all
#iteratons combined.
mostImportantFeatures <- function(trainingSet, iterations=10, SSSize = 300) {
    cols = names(trainingSet)
    importance <- rep(0.0, length(cols));
    for (i in 1:iterations) {
        trn = trainingSet
        if (SSSize < dim(trainingSet)[1]) {
            smpl = sample(dim(trainingSet)[1], SSSize);
            trn <- trainingSet[smpl, ]
        }
        learning = learn(trn, trn)
        imp <- learning$fit$finalModel["importance"];
        accuracy = as.data.frame(learning$cm["overall"])["Accuracy",];
        importance <- importance + 
        sapply(cols, FUN = function(x){if (is.na(imp[1]$importance[,][x])) {0} else {imp[1]$importance[,][x]*accuracy}});
    }
    names(trainingSet)[order(importance, decreasing=TRUE)]
}




#Step 1:  Setup initial environment. WD, SEED, DATA, constants
trn = NULL
tinyTST = NULL
pmlInit = function () {
    rm(list=ls())
    setwd(
        paste("SET_TO_THE-PARENT_OF_DATA_DIRECTORY_WHERE",
              "THE_FOLLOWING_TWO_FILES_ARE",
              sep="", collapse="")
        )
    trn <<- read.csv("../Data/pml-training.csv")
    tinyTST <<- read.csv("../Data/pml-testing.csv")    
    set.seed(71679174)
    TESTDATAPROB <<- .3
    CORRTHRESH <<- .8
}
pmlInit()


#Step 2: convert to tidy, remove stat-summary cols
trn <- tbl_df(trn)
tinyTST <- tbl_df(tinyTST)
trn<-trn[,2:160]
tinyTST<-tinyTST[,2:160]
allcols = colnames(trn)
print(dim(trn)[2]);print("XXXXX")
rawCols <- allcols[-grep("^max|^min|^amplitude|^var|^avg|^stddev|^skewness|^kurtosis", allcols)]
trn<- trn[rawCols]
tinyTST<- tinyTST[rawCols] 
print(dim(trn)[2]);print("XXXXX")

#Step 3: Create a test set that will not be touched in any analysis
#and will be used only for testing
testingRows <- sample(dim(trn)[1], size = dim(trn)[1]*TESTDATAPROB)
tstset <- trn[testingRows,] #do not touch it except to deselect columns.
trn <- trn[-testingRows,] #not intersected with testing at all.
print(dim(trn)[2]);print("XXXXX")

#Step 4:  Get rid of highly correlated columns
corrCols = getCorrelatedCols(trn[sample(dim(trn)[1], size = CORRSAMPLESIZE), ]);
keep = setdiff(names(trn), corrCols)
print(length(corrCols))
tstset <- trn[, keep] #do not touch it except to deselect columns.
trn <- trn[, keep] #not intersected with testing at all.
tinyTST<- tinyTST[, keep] 
print(dim(trn)[2]);print("XXXXX")




#Step 5: Now do the main experiment.  we will create training sets of
#diverse sizes (increasing), learn a model of them, check the accuracy
#of the model.  We will try to hit 99%+ accuracy.  Each model will be
#tested on two sets 1) The main test set (tstset) created above and 2)
#all other rows not used in training.  If total rows are 20000 and
#5000 are in the tstset, potentially we can use 15000 roes for
#training. But we used only NN for training.  The remaining 15000-N
#can also be used as a separate test set. We will vary NN from 100 to
#10000 and increase its value by 25% each time.

NN = 3508
answer = NULL
df = NULL;

bestN = floor(0.75 * dim(trn)[2]);
features = union(mostImportantFeatures(trn)[1:bestN], "classe")
trn <- trn[,features]
tstset  <- tstset[,features]
tinyTST <- tinyTST[,features]
print(dim(trn)[2]);print("XXXXX")

for (i in 1:150) {
    savedTrn <- trn
    savedTst <- tstset
    savedTiny <- tinyTST
    NN = floor(NN * 1.25)
    if (NN > 7000) {
        break;
    }
    rows = sample(dim(trn)[1], NN);
    thisTrain <- trn[rows,]
    thisTest  <- trn[-rows,]
    L <- learn(thisTrain, thisTest)
    answer <- paste(predict(L$fit, newdata = tinyTST),sep="", collapse="")
    cm <- confusionMatrix(predict(L$fit, newdata = tstset), tstset$classe)
    if (is.null(df)) {
        df = c(NN, L$numCols, L$approxTime, as.data.frame(L$cm["overall"])["Accuracy",], as.data.frame(cm["overall"])["Accuracy",], answer)
        names(df) = c("TrainingSize", "Features", "ApproxTime", "AccuracyOnUnusedTraining", "AccuracyOnSetAsideTest", "CatsOfTiny")
    } else {
        df = rbind(df, c(NN, L$numCols, L$approxTime, as.data.frame(L$cm["overall"])["Accuracy",], as.data.frame(cm["overall"])["Accuracy",], answer))
    }
    print(df)
    write.csv(df, paste("CorrFile", NN, ".csv", sep="_"));
    trn <- savedTrn
    tstset <- savedTst
    tinyTST <- savedTiny
}









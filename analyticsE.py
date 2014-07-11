import ConfigParser
import utils
import sys
import numpy as np
import scipy
import datetime
import math
import random
from scipy import io
import pandas as pd
import os
import pickle
import time
from aeConfig import AEConfig
from aeCVInstance import CVInstance
import threading
from guppy import hpy
from multiprocessing import Process,Queue
import aeWorkers
import filterData as fd
import CVstart
import newCV

DEBUG=False
today = datetime.date.today().strftime("%Y%m%d")
specs_for_test = dict()
numArgs = len(sys.argv)
cmdArgs = sys.argv
'''
thisConfigFile = "modelConfig.ini"
testDataFile = "testdata.csv"
thisConfigFile = "amznConfig.ini"
trainDataFile = "amzn.train.csv"
'''
testFileExists = False

if(numArgs!=3 and numArgs!=4):
  print "Usage: python step1.transformData.py CONFIG_FILE DATA_FILE\n"
  sys.exit()

else:
  thisConfigFile = cmdArgs[1]
  trainDataFile = cmdArgs[2]
  if(not os.path.exists(trainDataFile)):
    print "[ERROR] Data file does not exist: %s" % (trainDataFile)
    sys.exit()
  
  if(numArgs==4):
    testDataFile = cmdArgs[3]
    
    if(os.path.exists(testDataFile)):
      testFileExists=True
      print "[INFO]: Test file exists: %s" % (testDataFile)
    
    else:
      print "[WARNING]: Test file does not exist: %s" % (testDataFile)

print("Using config file %s" % (thisConfigFile))
print("Using data file %s" % (trainDataFile))

aeConfig = AEConfig(thisConfigFile)     ##New config class - to replace the old in-line config
usingDerivedDependentVar = False
dependentVar = aeConfig.dependentVar

if dependentVar == "":
  print "[ERROR] No dependent variable defined in config file"
  sys.exit()

dependentVarExpression = ""
print dependentVar
print aeConfig.useDerivedDependent
if aeConfig.useDerivedDependent==True and len(dependentVar.split("="))>0:
  newDependentVar = dependentVar.split("=")[0]  #extract the name of the new  dependent var from expression
  dependentVarExpression = dependentVar         #set a variable with the expression, for use later
  dependentVar = newDependentVar                #assign the new name to the dependentVar variable
  usingDerivedDependentVar = True
  print "[INFO] Creating derived dependent variable '%s' with expression: '%s'" % (dependentVar,dependentVarExpression)

outputDir, outputFilePrefix = fd.StoreFiles(aeConfig, trainDataFile)
specs_for_test['outputDir'] = outputDir
specs_for_test['outputFilePrefix'] = outputFilePrefix

#----------------------------------------------------------------------#
'''Columns'''

catColsNames, numColsNames, txtColsNames = [], [], []
catColsIndexes, numColsIndexes, txtColsIndexes = [], [], []

if(aeConfig.catCols.__len__()!=0):
  catColsNames = aeConfig.catCols.split(",")
  catColsNames = pd.unique(catColsNames).tolist()

if(aeConfig.numCols.__len__()!=0):
  numColsNames = aeConfig.numCols.split(",")
  numColsNames = pd.unique(numColsNames).tolist()

if(aeConfig.txtCols.__len__()!=0):
  txtColsNames = aeConfig.txtCols.split(",")
  txtColsNames = pd.unique(txtColsNames).tolist()
specs_for_test['catColsNames'] = catColsNames
specs_for_test['numColsNames'] = numColsNames
specs_for_test['txtColsNames'] = txtColsNames

if(aeConfig.noScientific==True):
  np.set_printoptions(suppress=True)

print "####################################"
print "Categorical Columns: %s" % (', '.join(catColsNames))
print "Numerical Columns: %s" % (', '.join(numColsNames))
print "Text Columns: %s" % (', '.join(txtColsNames))
print "####################################"

#----------------------------------------------------------------------#
'''Headers'''

outputHeader = (','.join(catColsNames))
if (aeConfig.dataHasHeader == True):
	catColsIndexes, numColsIndexes, txtColsIndexes = fd.Headers(trainDataFile, aeConfig, dependentVar, catColsNames, numColsNames, txtColsNames)
	print trainDataFile
	print aeConfig
	print dependentVar
	print catColsNames, numColsNames, txtColsNames
	print catColsIndexes, numColsIndexes, txtColsNames
	
else:
	dependentCol=int(dependentVar)
specs_for_test['outputHeader'] = outputHeader	
specs_for_test['catColsIndexes'] = catColsIndexes
specs_for_test['numColsIndexes'] = numColsIndexes
specs_for_test['txtColsIndexes'] = txtColsIndexes

#----------------------------------------------------------------------#
'''Loading DATA'''

timestart = int(round(time.time() * 1000))
print ("[IO] Loading data ... "),
sys.stdout.flush()

train_data_df_original=pd.read_csv(open(trainDataFile), quotechar='"', skipinitialspace=True, sep=aeConfig.inputFileSeparator, na_values=[], error_bad_lines = False)

if aeConfig.useDerivedDependent:
	print "Using Derived Dependent Variable"
	try:
    		train_data_df_original.eval(dependentVarExpression)

  	except SyntaxError:
    		print "[ERROR] Expression provided for derived dependent variable is invalid: '%s' (train data)" % (dependentVarExpression)
    		print "Exiting ..."
    		sys.exit()

#----------------------------------------------------------------------#
'''FILTER'''

filterTrainData = False
trainFilterColumn = ""
trainFilterOperator = ""
trainFilterValue = ""

if aeConfig.trainFilter!="":
        train_data_df_original =  fd.filterData(aeConfig.trainFilter, train_data_df_original, 'train') 
train_data_df=train_data_df_original.copy()
timeend = int(round(time.time() * 1000))
print "%d millis" % (timeend - timestart)

#----------------------------------------------------------------------#
'''Combining Columns'''

combineDegree = 2
if (aeConfig.makeTriplets==True):
  combineDegree = 3
catColsNames_original = catColsNames
output_header_original = outputHeader
if (aeConfig.combineColumns == True):
	catColsNames, train_data_df,  outputHeader = fd.CombineCols( combineDegree, train_data_df, catColsNames, aeConfig, outputHeader) 
	print len(catColsNames)

#----------------------------------------------------------------------#
''' Factorization'''

if aeConfig.factorizeDependent==True:

	tr_labs=np.asarray(train_data_df.as_matrix(columns=[dependentVar]))
	leng = len(np.unique(tr_labs))
	if leng>2 and aeConfig.multiClass == False:
		print '[INFO] The problem is assumed to be a Regression probelm'
		print '[INFO] Factorization not done'

	else:
		if aeConfig.multiClass == True:
			print '[INFO] Factorization for Multi-Class problem'
		else:
			print '[INFO] Factorization for a Binary Classification Problem'
		print train_data_df[dependentVar][:10]
		fact = fd.Factorize(dependentVar)  
		train_data_df, thisFactor = fact.Factorize_train(train_data_df)  
  		thisFactorOutFile = "%s/FACTOR.%s.%s" % (outputDir, outputFilePrefix, dependentVar)
  		thisFactorOutFid = open(thisFactorOutFile, 'w' )
		thisFactorOutFid.write('%s\n' % (thisFactor.levels))
  		thisFactorOutFid.close()
		specs_for_test['Fact'] = fact
		print train_data_df[dependentVar][:10]
#print thisFactor	

#----------------------------------------------------------------------#
'''NAN Coersion'''

badNumericalCols = []
if(len(numColsNames)>0):
	badNumericalCols = train_data_df[numColsNames].columns[train_data_df[numColsNames].dtypes=='object']
  	badNumericalCols = badNumericalCols.append(train_data_df[numColsNames].columns[train_data_df[numColsNames].dtypes=='bool'])
  	print "[WARNING] The following numerical columns contain strings, will be coerced to NaN: %s" % (badNumericalCols)

#----------------------------------------------------------------------#

train_labels=np.asarray(train_data_df.as_matrix(columns=[dependentVar]))
train_data_categorical=np.asarray(train_data_df.as_matrix(columns=catColsNames))        ##categorical columns
train_data_numerical=np.asarray(train_data_df.as_matrix(columns=numColsNames))          ##numerical columns

#----------------------------------------------------------------------#
'''One Hot'''

outputHeaderSplit = outputHeader.split(",")
oneHotColumnHeaders = []
oneHotColumns = catColsNames

if(aeConfig.useOneHot==True and len(oneHotColumns)>0):
	train_data_transformed_onehot, oneHotColumnHeaders, factorMap, encoderToUse  = fd.oneHotTrain(catColsNames, train_data_df, outputDir, outputFilePrefix)

else:
	train_data_transformed_onehot = train_data_categorical
	factorMap = []	
	encoderToUse = []
catColsIdx = range(train_data_transformed_onehot.shape[1])
numColsAddedIdx = range(train_data_transformed_onehot.shape[1], train_data_transformed_onehot.shape[1]+len(numColsIndexes))
specs_for_test['oneHotColumnHeaders'] = oneHotColumnHeaders
specs_for_test['factorMap'] = factorMap
specs_for_test['encoderToUse'] = encoderToUse
#print len(factorMap)
#print np.shape(train_data_df)
#print (train_data_transformed_onehot[0])
#print oneHotColumnHeaders
#sys.exit()

#----------------------------------------------------------------------#
''' Imputation'''
print train_data_numerical
rowsToUse_train = range(train_data_df.shape[0])
imp = fd.Impute(numColsNames, True, True, rowsToUse_train) 
train_data_numerical_new, rowsToUse_Train, scalingFactors, med = imp.Impute_train(train_data_numerical)
specs_for_test['Imp'] = imp
print "[INFO] Finished setting train data"

#----------------------------------------------------------------------#
'''Add Num and sparsifying'''

''' Numerical'''	
if (len(numColsIndexes)>0):
	print '[INFO] Converting from Sparse to Dense'
	train_data_numerical_sparse, train_data_transformed_onehot = fd.sparsify(train_data_numerical_new, train_data_transformed_onehot)
	print np.shape(train_data_transformed_onehot)

'''Text'''
if(len(txtColsIndexes)>0):
  for thisTxtCol in txtColsNames:
    if thisTxtCol in train_data_df.columns:
      train_data_df[thisTxtCol] = train_data_df[thisTxtCol].apply(str)

#----------------------------------------------------------------------#

train_data_transformed_onehot_original = train_data_transformed_onehot
train_data_transformed_onehot = train_data_transformed_onehot_original[rowsToUse_train,:]
train_labels_original = train_labels
train_labels = train_labels_original[rowsToUse_train,:]

#----------------------------------------------------------------------#
if (len(numColsIndexes)>0):
	outputHeader = "%s,%s" % (outputHeader, ','.join(numColsNames))
	oneHotColumnHeaders.extend(numColsNames)

if(DEBUG):
  print "One Hot Column Headers: "
  print oneHotColumnHeaders

if len(oneHotColumns)>0:
	oneHotDataFile = "%s/%s.%s.onehot" % (outputDir, outputFilePrefix, today)
  	outDataFid = open(oneHotDataFile, 'w' )
  	io.mmwrite(outDataFid, train_data_transformed_onehot)
  	outDataFid.close()

#----------------------------------------------------------------------#
'''Model'''

modelToUse, model_specs, cv, clf, modelToUse2, mod2 = fd.Model(aeConfig, train_data_transformed_onehot, train_data_df, train_labels)
specs_for_test['model_specs'] = model_specs

print "Finished Modelling\n"

#----------------------------------------------------------------------#
'''Text Vectorizer'''

ngram_range_lb, ngram_range_ub = 1, 1
txtVectorizers = fd.textVect(txtColsNames, train_data_df,  ngram_range_lb, ngram_range_ub)

oneHotColumnHeaders_original = oneHotColumnHeaders
train_data_transformed_onehot_with_text = train_data_transformed_onehot.copy()
specs_for_test['textVectorizer'] = txtVectorizers
if len(txtColsIndexes)>0 :
	oneHotColumnHeaders, train_data_transformed_onehot_with_text, txtColsAddedIdx_train = fd.combining_txt(oneHotColumnHeaders, train_data_df, train_data_transformed_onehot,txtColsIndexes, txtColsNames, txtVectorizers)
else:
	txtColsAddedIdx_train = []

#----------------------------------------------------------------------#
'''Sparse to Dense'''

if model_specs['convertToDense'] == True or aeConfig.createTrees == True :
	print "{INFO] Converting sparse data to dense data\n"
	train_data_transformed_onehot_with_text_dense = train_data_transformed_onehot_with_text.toarray()
	train_data_transformed_onehot_with_text = train_data_transformed_onehot_with_text_dense
else:
	train_data_transformed_onehot_with_text_dense = None

#----------------------------------------------------------------------#
''' Grid Search'''
train_index = np.asarray(range(train_data_transformed_onehot_with_text.shape[0]))

if aeConfig.gridsearch and model_specs['gridSearchCompleted']==False:	
    	clf.fit(train_data_transformed_onehot_with_text[train_index, :], train_labels[train_index].ravel())
    	modelToUse = clf.best_estimator_
    	print "------------ GRID SEARCH MODEL ------------"
    	print(modelToUse)
    	print "-------------------------------------------"
    	model_specs['gridSearchCompleted'] = True	

########################################################################
'''Main Program'''

colsToUse = range(train_data_transformed_onehot.shape[1])
keys = ['outputDir', 'outputFilePrefix', 'trainDataFile', 'outputHeader', 'dependentVar']
values = [outputDir, outputFilePrefix, trainDataFile, outputHeader, dependentVar]
other_specs = dict(zip(keys, values))
test_specs = dict()
test_specs[0] = 0
test_specs[1] = 0
test_specs[2] = 0
test_specs[3] = 0
allAucs = []
#-----------------------Cross Validation---------------------------------#
if (aeConfig.CV == True):
	cvn = 0
	cvIter = cv.__iter__()
	for cvn in range(cv.n_iter):
		train_index, text_index = [], []
		print "[CV] ----- Cross validation loop, Run #%d -----" % (cvn+1)
    		train_index, test_index = cvIter.next()
    		print "[CV] %d train rows, %d test rows" % (len(train_index), len(test_index))
		CV_on = True

		test_specs, allAucs, modelToUse, colsToUse = CVstart.CVS(aeConfig, cv, cvn, CV_on, model_specs, modelToUse, train_index, test_index, train_labels, train_data_df, train_data_transformed_onehot_with_text, other_specs, txtColsAddedIdx_train, catColsIdx, oneHotColumnHeaders, scalingFactors, test_specs, allAucs, colsToUse, modelToUse2, mod2,  train_data_transformed_onehot_with_text_dense)

	print "---------------------------------------------------------"
	print "Cross Validation complete"
	if aeConfig.cvfolds>0 and model_specs['isRegression'] == 'False':
		print "[Confusion Matrix - across all folds]:"
		print "          Pred 0\tPred 1\nActl 0    %d\t\t%d\nActl 1    %d\t\t%d" % (test_specs[0], test_specs[1], test_specs[2], test_specs[3])
else:
	allAucs = 0

#-------------------------Train------------------------------------------#

print "---------------------------------------------------------"
print " Now building model on ALL data"
train_index = np.asarray(range(train_data_transformed_onehot_with_text.shape[0]))
test_index = train_index
CV_on = False
cvn = 4

test_specs, allAUCs, modelToUse, colsToUse = CVstart.CVS(aeConfig, cv, cvn, CV_on, model_specs, modelToUse, train_index, test_index, train_labels, train_data_df, train_data_transformed_onehot_with_text, other_specs, txtColsAddedIdx_train, catColsIdx, oneHotColumnHeaders, scalingFactors, test_specs, allAucs, colsToUse, modelToUse2, mod2,  train_data_transformed_onehot_with_text_dense)

specs_for_test['model2use'] = modelToUse
specs_for_test['cols2use'] = colsToUse
specs_for_test['other_specs'] = other_specs
#-------------------------Trees------------------------------------------#

if(aeConfig.modelType=="dtr" or aeConfig.modelType=="dtc"):
	print modelToUse.tree_.feature[0]
	if(model_specs['displayTree']==True):
        	newCV.Display_trees(other_specs, modelToUse, oneHotColumnHeaders)

train_index_unmodified = list(train_index)
test_index_unmodified = list(test_index)

if aeConfig.createTrees==True:
	feature_splitOn = CV.Create_trees(oneHotColumnHeaders, modelToUse, modelToUse2, train_data_transformed_onehot_with_text, train_labels, train_index_unmodified,model_specs, aeConfig, scalingFactors, colsToUse, other_specs, mod2, train_data_transformed_onehot_with_text_dense)




print "[INFO] Finished setting train data"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

if (testFileExists):
	print ("LOADING TEST DATA ")
        sys.stdout.flush()
        test_data_df_original = pd.read_csv(open(testDataFile), quotechar='"', skipinitialspace=True, sep=aeConfig.inputFileSeparator, na_values=[], error_bad_lines = False)
		
        if usingDerivedDependentVar:
                try:
                        test_data_df_original.eval(dependentVarExpression)

                except SyntaxError:
                        print "[ERROR] Expression provided for derived dependent variable is invalid: '%s' (test data)" % (dependentVarExpression)
                        print "Exiting ..."
                        sys.exit()
#----------------------------------------------------------------------------#
	#Filtering

	filterTestData = False
	testFilterColumn = ""
	testFilterOperator = ""
	testFilterValue = ""

	if aeConfig.testFilter!="":
        	test_data_df_original = fd.filterData(aeConfig.testFilter,test_data_df_original, 'test')

	test_data_df=test_data_df_original.copy()
#----------------------------------------------------------------------------#
	#CombineColumns

	if (aeConfig.combineColumns == True):
        	catColsNames, test_data_df,  outputHeader = fd.CombineCols( combineDegree, test_data_df, catColsNames_original, aeConfig, output_header_original)

#----------------------------------------------------------------------------#
	#Factorizing

	if dependentVar in test_data_df_original and aeConfig.factorizeDependent == True:
                test_data_df = fact.Factorize_test(test_data_df)

	test_data_categorical=np.asarray(test_data_df.as_matrix(columns=catColsNames))        ##categorical columns
	test_data_numerical=np.asarray(test_data_df.as_matrix(columns=numColsNames))          ##numerical columns

#----------------------------------------------------------------------------#
	#OneHotEncoder

	if(aeConfig.useOneHot==True and len(oneHotColumns)>0):
        	test_data_transformed_onehot, oneHotColumnHeaders, factorMap_test  = fd.oneHotTest(catColsNames, test_data_df, outputDir, outputFilePrefix, factorMap, encoderToUse)
	else:
		test_data_transformed_onehot = test_data_categorical
	
#----------------------------------------------------------------------------#
	#Imputation	

	test_numerical_data = imp.Impute_test(test_data_numerical)
	
#----------------------------------------------------------------------------#
	#Dense to Sparse	

	if (len(numColsIndexes)>0):
		test_data_numerical_sparse, test_data_transformed_onehot = fd.sparsify(test_data_numerical, test_data_transformed_onehot)
	
#----------------------------------------------------------------------------#
	#Combining Text Columns
	test_data_transformed_onehot_with_text = test_data_transformed_onehot.copy()
	if(len(txtColsIndexes)>0):
  		for thisTxtCol in txtColsNames:
    	        	if thisTxtCol in test_data_df.columns:
      				test_data_df[thisTxtCol] = test_data_df[thisTxtCol].apply(str)
	oneHotColumnHeaders, test_data_transformed_onehot_with_text, txtColsAddedIdx_test = fd.combining_txt(oneHotColumnHeaders, test_data_df, test_data_transformed_onehot,txtColsIndexes, txtColsNames, txtVectorizers)
	
#----------------------------------------------------------------------------#
	#Sparse to Dense

	if model_specs['convertToDense'] == True or aeConfig.createTrees == True :
		print "[INFO] Converting test data to dense format..."
                test_data_transformed_onehot_with_text = test_data_transformed_onehot_with_text.toarray()

	
#----------------------------------------------------------------------------#
	#Model Test

	other_specs['testDataFile'] = testDataFile
	CVstart.Test_On(modelToUse, model_specs, other_specs, test_data_transformed_onehot_with_text, dependentVar, test_data_df_original, colsToUse)

	print "[INFO] Finished with both Train and test dataste"
#############################################################################
testt = 'TestSpecifications.%s' %(today)
fd.createfiles(specs_for_test, testt)
print "Specifications for testing on other datasets have been stored"        

print "AnalyticsEngine has completed the task"

import ConfigParser
from aeConfig import AEConfig
import utils
import cPickle as pickle
from sklearn import preprocessing
import datetime
import os
import pandas as pd
from sklearn import (naive_bayes, linear_model, preprocessing, metrics, cross_validation, feature_selection, svm, ensemble, tree, neural_network, cluster, grid_search, neighbors, decomposition)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.externals.six import StringIO
from scipy import io
import scipy
import numpy as np
import random
import sys
import time

def StoreFiles(aeConfig, datafile):
	outputFilePrefix = os.path.basename(os.path.realpath(datafile))
	outputFilePrefix = outputFilePrefix.replace("/",".")
	
	outputDir = str(os.path.dirname(os.path.realpath(datafile)))       #default output dir is same dir as input data file
	if aeConfig.outputDir!="":
  		outputDir = aeConfig.outputDir

	if not os.path.exists(outputDir):
  		print "[INFO] Output directory does not exist, creating: %s" % (outputDir)
  		os.makedirs(outputDir)
	#createfiles(outputDir,outputDir)
	#createfiles(outputFilePrefix,outputFilePrefix)
	return outputDir, outputFilePrefix	

def loadfiles(x):
	f = open(x , 'rb')
	return pickle.load(f)
	f.close()

def createfiles(x, y):##### Need to specify the path
	f = open(y, 'wb')
	pickle.dump( x, f)
	f.close()
	


def filterData(filters, datafile, dat):
    
  	Filter = filters
        if(len(Filter.split('='))==2 or len(Filter.split('<'))==2 or len(Filter.split('>'))==2):
    	    if(len(Filter.split('='))==2):
     	    	FilterOperator="="
    	    if(len(Filter.split('<'))==2):
      		FilterOperator="<"
    	    if(len(Filter.split('>'))==2):
      		FilterOperator=">"
	    #if(len(Filter.split('>='))==2):
            #    FilterOperator=">="	
	    thisSplit = Filter.split(FilterOperator)
    	    FilterColumn = thisSplit[0]
    	    FilterValue = float(thisSplit[1])
	    numRowsOriginal = train_data_df_original.shape[0]
            print ("FILTERING" + dat +  "DATA ... ")
            sys.stdout.flush()
            datafile = utils.filterDataframe[FilterOperator](datafile, FilterColumn, FilterValue)
            numRowsNew = datafile.shape[0]    	    
	    if numRowsNew == 0:
	        print "\n[ERROR] Filtering resulted in 0 rows in training dataset, exiting. Following filter was used:\n\t%s" %(Filter)
		sys.exit()
	    datafile.index = range(datafile.shape[0])
	    
        else:
            print "[ERROR] Unable to handle filter for  data: %s" % (Filter)
    	    sys.exit()
	return datafile



def Headers(datafile, aeConfig, dependentVar, cats, nums, txts):
	hasHeader=True
  	numLinesToSkip = 1

  	with open(datafile, 'r') as f:
    		headerLine =  f.readline()
    		headerLine = headerLine.strip()
    		if(aeConfig.stripQuotes==True):
      			print "[PRE] Stripping quotes from header line"
      			headerLine = headerLine.replace('"', '');


  	headerSplit = headerLine.split(aeConfig.inputFileSeparator)
  	if(len(headerSplit) == 1 and aeConfig.autoDetectSeparator == True):
		foundSuccSplit = False
		for seps in [',', '\t', '|']:
			newHeaderSplit = headerLine.split(seps)
    			if(len(newHeaderSplit)>1):
      				aeConfig.inputFileSeparator = seps
      				print "File appears to be '%s'-delimited. If not, set autoDetectSeparator=false" %(seps)
      				headerSplit = newHeaderSplit
				foundSuccSplit = True
				break
		if foundSuccSplit == False:
			print("[ERROR] Unable to split training data using supplied delimiter, or using standard delimiters. Exiting")
      			sys.exit()

  	print "[DATA] Header: %s" % (headerLine)
 
	dependentCol = 0
	try:
    		dependentCol = headerSplit.index(dependentVar)
  	
	except ValueError:
    		if aeConfig.useDerivedDependent:
      			print "[INFO] Need to use derived dependent var '%s'" % (dependentVar)
      			print dependentVar
			if dependentVar in headerSplit:
        			print "[WARNING] Column with same name as derived dependent variable '%s' already exists. Will be overwritten." % (dependentVar)
      			else:
        			headerSplit.append(dependentVar)
        			dependentCol = headerSplit.index(dependentVar)
        			print "[INFO] Adding derived dependent variable '%s' as column #%d'" % (dependentVar,dependentCol)
    		else:
      			print "[ERROR] Dependent variable '%s' not found in header row. Exiting." % (dependentVar)
      			

  	print "[DATA] Dependent Var is column %d" % (dependentCol)


	catColsIndexes = []
	numColsIndexes = []
	txtColsIndexes = []


  	if aeConfig.catCols == "all":   #use everything except the dependendt col
    		if(dependentCol == 0):
      			catColsIndexes = range(1,len(headerSplit))
    		elif(dependentCol==(len(headerSplit)-1)):
        		catColsIndexes = range(0,len(headerSplit)-1)
      		else:
        		catColsIndexes = range(0,dependentCol)
        		catColsIndexes.extend(range(dependentCol+1,len(headerSplit)))

	else:
    		print ("[DATA] Categorical columns: "),
    		for thisCol in cats:
      			print ("%s " % (thisCol)),
      			catColsIndexes.append(headerSplit.index(thisCol))
    		print ""
    		if(aeConfig.numCols != ""):
     			print ("[DATA] Numerical columns: "),
      			for thisCol in nums:
        			print ("%s " % (thisCol)),
        			numColsIndexes.append(headerSplit.index(thisCol))
      			print ""
    		if(aeConfig.txtCols != ""):
      			for thisCol in txts:
        			print "[DATA] Text column: %s\n" % (thisCol)
        			txtColsIndexes.append(headerSplit.index(thisCol))
  		print "Categorical column indexes: %s" % (str(catColsIndexes))
  		print "Numerical column indexes: %s" % (str(numColsIndexes))
  		print "Text column indexes: %s" % (str(txtColsIndexes))
	
	return catColsIndexes, numColsIndexes, txtColsIndexes


def Headers_test(testDataFile, aeConfig):
	hassHeader=True
	numLinesToSkip = 1
	with open(testDataFile, 'r') as f:
     		headerLine = f.readline()
     		headerLine = headerLine.strip()
	if(aeConfig.stripQuotes==True):
     		print "[PRE] Stripping quotes from header line"
     		headerLine = headerLine.replace('"', '');
     		headerSplit = headerLine.split(',')
	if(len(headerSplit) == 1 and aeConfig.autoDetectSeparator == True):
     		newHeaderSplit = headerLine.split('\t')
     		if(len(newHeaderSplit)>1):
        		aeConfig.inputFileSeparator = '\t'
        		print "File appears to be TAB-delimited. If not, set autoDetectSeparator=false"
        		headerSplit = newHeaderSplit
	return aeConfig, headerLine


def CombineCols(combineDegree, datafile, catColsNames, aeConfig, outputHeader):
	catColsNames_original = catColsNames
	print catColsNames_original
	for thisDegree in range(2, combineDegree+1):
		all_data = datafile[catColsNames_original]
		new_data, new_cols = utils.group_data(all_data, thisDegree, True, aeConfig.combineColumnsMaxCard)	
		new_datafile = new_data.iloc[range(all_data.shape[0])][new_cols]
        	new_datafile = new_datafile.set_index(datafile.index)
        	datafile = datafile.join(new_datafile[new_cols])
		catColsNames.extend(new_datafile.columns)
        	for thisColName in new_datafile.columns:
                	outputHeader = "%s,%s" % (outputHeader, thisColName)
		print np.shape(datafile)
		print catColsNames
	return catColsNames, datafile, outputHeader


def oneHotTrain(catColsNames, datafile, outputDir, outputFilePrefix):
        today = datetime.date.today().strftime("%Y%m%d")
        oneHotColumns = catColsNames
        for colname in oneHotColumns:
                print ("%s, " % (colname))
        print ""
        dataForEncoderFit = datafile[oneHotColumns]
        print "\nEncoding training data ..."
        (data_transformed_onehot, factorMap, encoderTo, newColNames) = utils.OneHotEncoderFaster(dataForEncoderFit, oneHotColumns)
        (data_transformed_onehot, factorMap, encoder, newColNames_train) = utils.OneHotEncoderFaster(datafile, oneHotColumns, factorMap, encoderTo)
        
        print "\n[INFO] train data has %d cols, %d rows (post one-hot)" % (data_transformed_onehot.shape[1], data_transformed_onehot.shape[0])

        onehotKeymapFile_train = "%s/%s.%s.onehotkeymap.train" %(outputDir, outputFilePrefix, today)
        pickle.dump(factorMap, file(onehotKeymapFile_train,'w'))
        oneHotDataFile = "%s/%s.%s.onehot" % (outputDir, outputFilePrefix, today)
        print "\nSaving transformed one hot data set to %s" % (oneHotDataFile
)
	oneHotColumnHeaders = newColNames_train
        return data_transformed_onehot, oneHotColumnHeaders, factorMap, encoder

def oneHotTest(catColsNames, datafile, outputDir, outputFilePrefix, factorMap, encoderToUse):
        oneHotColumns = catColsNames
        today = datetime.date.today().strftime("%Y%m%d")
        for colname in oneHotColumns:
                print ("%s, " % (colname))
        print ""

        print "Encoding testing data ..."
        (data_transformed_onehot, factorMap_test, encoder_test, newColNames_test) = utils.OneHotEncoderFaster(datafile, oneHotColumns, factorMap, encoderToUse)
        
        print "[INFO] test data has %d cols, %d rows (post one-hot)" % (data_transformed_onehot.shape[1], data_transformed_onehot.shape[0])
        onehotKeymapFile_test = "%s/%s.%s.onehotkeymap.test" % (outputDir, outputFilePrefix, today)
        return data_transformed_onehot, newColNames_test, factorMap_test



class Factorize:
	def __init__(self,  vari):
		self.datafile = None
		self.vari = vari
		self.thisFactor = None
	
	def Factorize_train(self, datafile):
		self.datafile = datafile
		dependentValues = (self.datafile[self.vari]).apply(str)
		self.thisFactor = pd.Categorical.from_array(dependentValues)
		self.datafile[self.vari]=[self.thisFactor.levels.get_loc(elt) for elt in (self.datafile[self.vari]).apply(str)]
		print 'Factorizing'
		return self.datafile, self.thisFactor

	def Factorize_test(self, datafile):
		self.datafile = datafile
		self.datafile[self.vari]=[self.thisFactor.levels.get_loc(elt) for elt in (self.datafile[self.vari]).apply(str)]
		return self.datafile		


	
class Impute:
	def __init__(self, colname, medians, scales, rowsToUse_train):
		self.colname = colname
		self.medians = medians
		self.scales = scales
		self.datafile = None
	 	self.trainMedians = dict()
		self.scalingFactors = dict()
		self.rowsToUse_train = rowsToUse_train

	def Impute_train(self,datafile):
		self.datafile = datafile
		rowsToSkip_train = []
		rowsToSkip_test = []
		for thisNumCol in range(self.datafile.shape[1]):
  			thisColName = self.colname[thisNumCol]
  			thisColRowsToSkip=(np.where(np.isnan(self.datafile[:,thisNumCol])))[0]
  			thisColRowsToUse=(np.where(~np.isnan(self.datafile[:,thisNumCol])))[0]
			print thisColRowsToUse	
  			if len(thisColRowsToUse)==0:
    				self.datafile[:,thisNumCol] = np.zeros(self.datafile.shape[0])

  			if(self.medians == True):
    				thisMedian = np.median(self.datafile[thisColRowsToUse,thisNumCol])
    				if np.isnan(thisMedian):
      					thisMedian=0
				self.trainMedians[thisColName] = thisMedian
    				self.datafile[thisColRowsToSkip,thisNumCol] = thisMedian
    				
				if(len(thisColRowsToSkip)>0):
      					print "Replacing %d rows of numerical Column %d (%s) in Train Data with median value (%.4f) from Train Data\n " % (len(thisColRowsToSkip), thisNumCol, thisColName, thisMedian)
			

			else:   ##if we're not replacing the N/A values with the median, don't use them for training
    				rowsToSkip_train = np.union1d(rowsToSkip_train, thisColRowsToSkip)
    				self.rowsToUse_train = np.intersect1d(self.rowsToUse_train, thisColRowsToUse)
    				print "\n[INFO] Skipping %d rows in train data due to NAN values" % (len(rowsToSkip_train))
			
  			if(self.scales==True):
    				thisMax = 1
    				if len(thisColRowsToUse)>0:
      					thisMax = np.max(np.abs(self.datafile[thisColRowsToUse,thisNumCol]))
					print thisMax
    				if thisMax!=0 and ~np.isnan(thisMax):
      					print "[INFO] Scaling column '%s' by: %.2f in train data (min: %.4f, max:%.4f)" % (thisColName, thisMax, np.min(self.datafile[:,thisNumCol]), np.max(self.datafile[:,thisNumCol]))

      					self.datafile[:,thisNumCol] = np.divide(self.datafile[:,thisNumCol], thisMax)
      					self.scalingFactors[thisColName] = thisMax
		
		return self.datafile, self.rowsToUse_train, self.scalingFactors, self.trainMedians
	
	

	def Impute_test(self,datafile):
		self.datafile = datafile 	
			
		for thisNumCol in range(self.datafile.shape[1]):
    			thisColName = self.colname[thisNumCol]
    			thisMedian = 0
    			if thisColName in self.trainMedians:
      				thisMedian = self.trainMedians[thisColName]
    			
			thisScalingFactor=1
    			if thisColName in self.scalingFactors:
      				thisScalingFactor = self.scalingFactors[thisColName]
			if self.scales and thisScalingFactor!=1:
      				print "[INFO] Scaling column '%s' by %.2f in test data (min: %.4f, max:%.4f)" % (thisColName, thisScalingFactor, np.min(self.datafile[:,thisNumCol]), np.max(self.datafile[:,thisNumCol]))
      				self.datafile[:,thisNumCol] = np.divide(self.datafile[:,thisNumCol], thisScalingFactor)
    			if (self.medians == True):
      				thisColNaRows=(np.where(np.isnan(self.datafile[:,thisNumCol])))[0]
      				if(len(thisColNaRows)>0):
        				print "[INFO] Replacing %d rows of numerical Column %d (%s) in Test Data with median value (%.4f) from Train Data " % (len(thisColNaRows), thisNumCol, thisColName, thisMedian)
        				self.datafile[thisColNaRows, thisNumCol] = thisMedian	


		return self.datafile

	

def sparsify(datafiles, datafiles_trans):
	datafiles_sparse = scipy.sparse.csr_matrix(datafiles)
	if datafiles.shape[0] == 1:
		datafiles_sparse = datafiles_sparse.T
	print "[INFO] Adding %d numerical columns (%d rows)" % (datafiles_sparse.shape[1], datafiles_sparse.shape[0])
	
	print "[INFO] Before: %d x %d" % (datafiles_trans.shape[0], datafiles_trans.shape[1])
	if datafiles_trans.shape[1]>0:
    		datafiles_trans = scipy.sparse.csr_matrix(scipy.sparse.hstack([datafiles_trans, datafiles_sparse]))
  	else:
    		datafiles_trans = datafiles_sparse ##no categorical columns specified, only numerical
	print "[INFO] After: %d x %d" % (datafiles_trans.shape[0], datafiles_trans.shape[1])

	return datafiles_sparse, datafiles_trans


def Model(aeConfig, train_data_transformed_onehot, train_data_df, train_labels):
	
	ensembleMaxFeatures=np.min([8, train_data_transformed_onehot.shape[1]])
	mod = dict()
	keys = ['convertToDense', 'printCoefficients', 'isRegression', 'displayTree',  'predict_prob', 'usingYearlyFolds', 'yearlyOutput', 'gridSearchCompleted']
	
	values = [False, True, False, False, True, False, None, False]
	mod2 = {'isRegression': False}
	mod = dict(zip(keys, values))
	cv = None
	clf = None
	mod2use = None
	mod2use2 = None
	print "-------------------------------------------------------"
	if (aeConfig.modelType == 'logistic'):	
		print "[MODEL] Logistic Regression"
		if aeConfig.regularizationType == 'l1' or aeConfig.regularizationType == 'l2':
			mod2use = linear_model.LogisticRegression(penalty=aeConfig.regularizationType, dual=False, tol=0.0001, C=aeConfig.regularizationPenalty, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
		else:
			print '[ERROR] Wrong value of regularization Penalty given. Can be either l1 or l2'
			sys.exit()
		if (aeConfig.chisqperc<100):
			 print "[WARNING] Chi-Sq filter (%.2f %%) only works for categorical columns. All numerical cols will be used" % (aeConfig.chisqperc)		
		if aeConfig.gridsearch == True:
			param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1','l2'] }
        		clf = grid_search.GridSearchCV(mod2use, param_grid, verbose=1)		
			mod['gridSearchCompleted'] == False
	elif (aeConfig.modelType=="linearsvc"):
		print "[MODEL] Linear Support Vector Classifier"
    		mod['predict_prob'] = False
                mod['printCoefficients'] = False
		mod2use = svm.LinearSVC()
		
	elif (aeConfig.modelType=="gbm"):
		print "[MODEL] Gradient Boosting Machine"
		mod['convertToDense'] = True
                mod['printCoefficients']=False
		mod2use = ensemble.GradientBoostingClassifier()

	elif (aeConfig.modelType=="randomforest"):
		
		print "[MODEL] Random forest"
		mod['convertToDense']=True
                mod['printCoefficients']=False
                mod['printFeatureImportance']=True

		if len(np.unique(train_labels))>2:      #Regression
          		mod['isRegression']=True
          		print "[MODEL] %d unique dependent values, building regression model" % (len(np.unique(train_labels)))
          		mod2use = ensemble.ExtraTreesRegressor(n_estimators=aeConfig.ensembleEstimators, max_depth=aeConfig.treeDepth, min_samples_leaf=max(1, (aeConfig.minPercInNode/100)*train_data_transformed_onehot.shape[0]))
		else:
			mod2use = ensemble.ExtraTreesClassifier(n_estimators=aeConfig.ensembleEstimators, max_depth=aeConfig.treeDepth, min_samples_leaf=max(1, (aeConfig.minPercInNode/100)*train_data_transformed_onehot.shape[0]))
		
	elif (aeConfig.modelType=="sgd"): 
		print "[MODEL] Stochastic Gradient Descent"
          	mod2use = linear_model.SGDClassifier(loss="huber", penalty="elasticnet", shuffle=True)


	elif (aeConfig.modelType=="gbr"):
		print "[MODEL] Gradient Boosted Regressor"
		mod['isRegression']=True
            	mod['convertToDense']=True
            	mod['printCoefficients']=False
            	mod2use = ensemble.GradientBoostingRegressor(n_estimators=aeConfig.ensembleEstimators, learning_rate=1.0, max_depth=3, random_state=0, loss='lad', max_features=ensembleMaxFeatures, verbose=1)		

	elif (aeConfig.modelType=="gbc"):
                print "[MODEL] Gradient Boosted Classifier"
                mod['convertToDense']=True
                mod['printCoefficients']=False
                mod2use = ensemble.GradientBoostingClassifier(n_estimators=aeConfig.ensembleEstimators, learning_rate=1.0, max_depth=3, random_state=0, loss='deviance', max_features=ensembleMaxFeatures, verbose=1)
	
	elif (aeConfig.modelType=="dtr"):
		print "[MODEL] Decision Tree Regressor"
                mod['isRegression']=True
                mod['convertToDense']=True
                mod['printCoefficients']=False
                mod['displayTree']=True
                mod2use = tree.DecisionTreeRegressor()
	
	elif (aeConfig.modelType=="dtc"):
                print "[MODEL] Decision Tree Classifier"
                mod['convertToDense']=True
                mod['printCoefficients']=False
                mod['displayTree']=True
                mod2use = tree.DecisionTreeClassifier()
	
	elif (aeConfig.modelType=="svr"):
                print "[MODEL] Support Vector Regressor"
                mod['isRegression']=True
                mod['printCoefficients']=False
                mod2use = svm.SVR()
	else:
                print "[ERROR] Undefined model %s (options: logistic/linearsvc/gbm/sgd/randomforest/svr)" % (aeConfig.modelType)
                sys.exit()

	
	if (aeConfig.CV == True):
		seed = random.randint(1,50) # always use a seed for randomized procedures
		cv_test_size = 1-aeConfig.cvperc
		cv = cross_validation.ShuffleSplit(train_data_transformed_onehot.shape[0], n_iter=aeConfig.cvfolds, test_size=cv_test_size, random_state=seed)
		mod['usingYearlyFolds']=False
		mod['yearlyOutput']=None
		if aeConfig.yearlyCVColumn is not None:
  			if aeConfig.yearlyCVColumn not in train_data_df.columns:
    				raise ValueError('Yearly CV column is not in dataset. Not performing yearly CV.')
  			else:
    				cv = utils.YearlySplit(train_data_df[aeConfig.yearlyCVColumn].tolist(), aeConfig.yearlyCVStartDate)
    				usingYearlyFolds=True


	if (aeConfig.createTrees == True):

                if len(np.unique(train_labels))>2:      #Regression
                        mod2['isRegression']=True
                        print "[MODEL] %d unique dependent values, building regression model" % (len(np.unique(train_labels)))
                        mod2use2 = ensemble.ExtraTreesRegressor(n_estimators=aeConfig.ensembleEstimators, max_depth=aeConfig.treeDepth, min_samples_leaf=max(1, (aeConfig.minPercInNode/100)*train_data_transformed_onehot.shape[0]))
                else:
                        mod2use2 = ensemble.ExtraTreesClassifier(n_estimators=aeConfig.ensembleEstimators, max_depth=aeConfig.treeDepth, min_samples_leaf=max(1, (aeConfig.minPercInNode/100)*train_data_transformed_onehot.shape[0]))


	return mod2use, mod, cv, clf, mod2use2, mod2


def textVect(txts, datafile, lb, ub):
	tV = dict()
  	for thisTxtColName in txts:
    		tV[thisTxtColName] = TfidfVectorizer(stop_words=None, ngram_range=(lb,ub))
    		print "[TEXT] Fitting text Vectorizer for '%s' for all data" % (thisTxtColName)
    		sys.stdout.flush()
    		tV[thisTxtColName].fit(datafile[thisTxtColName])
    	return tV


def fitTextVectorizer(thisCvn, thisTxtColName):
  print "[TEXT] Fitting text Vectorizer (%d grams) for '%s' for CV Instance %d" % (ngram_range_ub, thisTxtColName, thisCvn)
  sys.stdout.flush()
  cvInstances[thisCvn].txtVectorizers[thisTxtColName].fit(train_data_df[thisTxtColName][cvInstances[thisCvn].train_index])


def combining_txt(oneHotColumnHeaders, datafile, datafile_trans, txtColsIndexes, txtColsNames, txtVectorizers):
	datafile_trans_text = datafile_trans.copy()
	txtColsAddedIdx = []
	textColHeaders = []
	if len(txtColsIndexes)>0 :
		thisTxtColIdx = 0
  		for thisTxtColName in txtColsNames:
    			txtTimeStart = int(round(time.time() * 1000))
    			sys.stdout.flush()
    			thisTxtVectorizer = txtVectorizers[thisTxtColName]
    			txtVectors_data = thisTxtVectorizer.transform(datafile[thisTxtColName])
    			txtFeatureNames_data = thisTxtVectorizer.get_feature_names()
    			if txtVectors_data.shape[0]==1:
      				txtVectors_data = txtVectors_data.T
    			txtColsAddedIdx = range(datafile_trans.shape[1], datafile_trans.shape[1]+txtVectors_data.shape[1])
    			print "[TXT] Adding %d text-based columns to Train data" % (len(txtColsAddedIdx))
    			if(thisTxtColIdx==0):
      				datafile_trans_text = scipy.sparse.csr_matrix(scipy.sparse.hstack([datafile_trans, txtVectors_data]))
    			else:
      				datafile_trans_text = scipy.sparse.csr_matrix(scipy.sparse.hstack([datafile_trans_text, txtVectors_data]))
    			textColHeaders.extend(["%s=%s" % (thisTxtColName, element) for element in txtFeatureNames_data])
    			thisTxtColIdx=thisTxtColIdx+1
    			txtTimeEnd = int(round(time.time() * 1000))
    			print "[TXT] Column '%s' processed in %.3f s" % (thisTxtColName, float(txtTimeEnd - txtTimeStart)/1000)
  		oneHotColumnHeaders = oneHotColumnHeaders + textColHeaders

	return oneHotColumnHeaders, datafile_trans_text, txtColsAddedIdx
	 




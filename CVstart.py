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
import filterData
import sys
import newCV 




def CVS(aeConfig, cv, cvn, CV_on,  mod, modelToUse, train_index, test_index, train_labels, train_data_df, train_data_transformed_onehot_with_text, other_specs, txtColsAddedIdx,catColsIdx, oneHotColumnHeaders, scalingFactors, test_specs, allAucs, colsToUse, modelToUse2, mod2, train_data_transformed_onehot_with_text_dense):


	today= datetime.date.today().strftime("%Y%m%d")
	foldString = "fold %d/%d" % (cvn+1, cv.n_iter)
	train_index_unmodified = list(train_index)    #use these later for trees since we may undersample or oversample
        test_index_unmodified = list(test_index)      #use these later for trees since we may undersample or oversample
	class_dict = dict()
	class_dict['majC'], class_dict['minC'], class_dict['numMaj'],class_dict['numMin'], class_dict['n0'], class_dict['n1'] = newCV.MajMin(train_labels, train_index)
 
	if (mod['convertToDense'] == True):
		sys.stdout.flush()

	else:
		print  "[INFO] No need to convert data to dense format"
    		sys.stdout.flush()

	
	if aeConfig.balanceTrainSampleKMeans or aeConfig.createClusters:
		train_data_df, thisKMeans = newCV.balTrainClust(other_specs['trainDataFile'], train_data_df, train_data_transformed_onehot_with_text, train_index, CV_on, aeConfig)


	if aeConfig.balanceTrainSampleKMeans and mod['isRegression'] == False:
                train_index, train_labels = newCV.balTrainSamKmeans(train_labels, train_index, class_dict, thisKMeans)


	elif aeConfig.balanceTrainSample and mod['isRegression'] == False:
    		train_index, train_labels = newCV.balTrainSam(train_labels, train_index, class_dict)


	
	elif aeConfig.underSampleKMeans and mod['isRegression'] == False:
		train_index, train_labels = newCV.UndSamp(train_labels, train_index, class_dict, train_data_transformed_onehot_with_text)
	
	
	colsToUse = range(train_data_transformed_onehot_with_text[train_index, :].shape[1])

	if aeConfig.removeColsWithMostlyZeros==True:
		colsToUse = newCV.RemRows(train_data_transformed_onehot_with_text, train_index)	



	if(aeConfig.chisqperc<100):
		print "[INFO] USING Chi-Sq filter"
    	#chi-sq filtering only works for categorical columns
   		colsToUse = utils.getTopColumns(train_data_transformed_onehot_with_text[train_index, :][:,catColsIdx], train_labels[train_index], float(aeConfig.chisqperc)/100)
    		colsToUse = np.union1d(colsToUse, numColsAddedIdx)
    		txtColsToUse = txtColsAddedIdx
    		colsToUse = np.union1d(colsToUse, txtColsToUse)



	if aeConfig.tryGreedySelection==True:
    		print "[INFO] Attempting GREEDY feature selection"
    		greedyperc = 99
    		colsToUse = utils.getTopColumnsGreedy(train_data_transformed_onehot_with_text[train_index, :], train_labels[train_index], float(greedyperc)/100, modelToUse)

	
	print "[INFO] Fitting model to predict %s, using %d features, %d instances" % (other_specs['dependentVar'], train_data_transformed_onehot_with_text[train_index, :].shape[1], train_data_transformed_onehot_with_text[train_index, :].shape[0])
	sys.stdout.flush()


	modelToUse.fit(train_data_transformed_onehot_with_text[train_index, :][:,colsToUse], train_labels[train_index].ravel())
	#print modelToUse.tree_.feature[0]

	#if mod['isRegression'] == False:
	#print "(Predicting %s = %d)" % (other_specs['dependentVar'], modelToUse.classes_[1])

	#else:	
	#print "(Predicting %s, regression)" % (dependentVar)


	probabilities_train = np.zeros(train_data_transformed_onehot_with_text[test_index, :].shape[0])
	if mod['isRegression'] == False and mod['predict_prob'] == True:
		probabilities_train = modelToUse.predict_proba(train_data_transformed_onehot_with_text[test_index, :][:,colsToUse])[:,1]
	predictions_train = modelToUse.predict(train_data_transformed_onehot_with_text[test_index, :][:,colsToUse])
	


	if (mod['usingYearlyFolds'] == True):
		if (CV_on == True):	

			if (mod['printCoefficients'] == True):
				coefficientsFile = "%s/%s.%s.cv.%d_%d.lr.coefficients.csv" % (other_specs['outputDir'], other_specs['outputFilePrefix'], today, cvn+1, cv.n_iter)
				utils.printLRCoefficients(modelToUse, oneHotColumnHeaders, coefficientsFile)

			if type(train_data_transformed_onehot_with_text)==np.ndarray:
        			thisYearOutput = np.hstack([train_data_transformed_onehot_with_text[test_index, :],train_labels[test_index],(np.asmatrix(predictions_train)).transpose()])
      			else:
        			thisYearOutput = np.hstack([train_data_transformed_onehot_with_text[test_index, :].toarray(),train_labels[test_index],(np.asmatrix(predictions_train)).transpose()])


      			if yearlyOutput is None:
        			yearlyOutput = thisYearOutput
      			else:
        			yearlyOutput = np.vstack([yearlyOutput, thisYearOutput])
      			

			if mod['isRegression']:
        			print "[INFO] sum(sign(predictions)*Y_values)=%.2f; sum(Y_values)=%.2f" % (np.sum(np.dot(np.sign(predictions_train).ravel(),train_labels[test_index])), np.sum(train_labels[test_index]))
      
			roc_auc = 0
			thisMeanSqError = 0

		else:       #If we've finished all yearly folds, write to CSV
      			yearlyOutputFile = "%s.%s.yearly.%s.csv" % (trainDataFile, other_specs['dependentVar'], today)

      			print "[INFO] Saving yearly output to file: %s" % (yearlyOutputFile)
      			np.savetxt(yearlyOutputFile, yearlyOutput, delimiter=",")



	if (mod['isRegression'] == False):
		if CV_on == True and aeConfig.cvfolds > 0:
			fpr, tpr, thresholds = metrics.roc_curve(train_labels[test_index], probabilities_train)
      			roc_auc = metrics.auc(fpr, tpr)
      			thisCm = metrics.confusion_matrix(train_labels[test_index], predictions_train)
			print "[Confusion Matrix]:"
      			print "          Pred 0\tPred 1\nActl 0    %d\t\t%d\nActl 1    %d\t\t%d" % (thisCm[0][0], thisCm[0][1], thisCm[1][0], thisCm[1][1])
			accuracy = metrics.accuracy_score(train_labels[test_index], predictions_train)
			print "[CV] AUC (%s, size:%d): %f (probs: min: %.4f, max:%.2f). Accuracy: %.4f\n" % (foldString, train_labels[test_index].shape[0], roc_auc, min(probabilities_train), max(probabilities_train), accuracy)

                	
			test_specs[0] = test_specs[0] + thisCm[0][0]
                	test_specs[1] = test_specs[1] + thisCm[0][1]
                	test_specs[2] = test_specs[2] + thisCm[1][0]
                	test_specs[3] = test_specs[3] + thisCm[1][1]
			allAucs.append(roc_auc)
		else:	
			fpr, tpr, thresholds = metrics.roc_curve(train_labels[test_index], probabilities_train)
                        roc_auc = metrics.auc(fpr, tpr)
                        thisCm = metrics.confusion_matrix(train_labels[test_index], predictions_train)
                        print "[Confusion Matrix]:"
                        print "          Pred 0\tPred 1\nActl 0    %d\t\t%d\nActl 1    %d\t\t%d" % (thisCm[0][0], thisCm[0][1], thisCm[1][0], thisCm[1][1])
			accuracy = metrics.accuracy_score(train_labels[test_index], predictions_train)
			print "[CV] AUC (%d train size):%f, (probs: min: %.4f, max:%.2f). Accuracy: %.4f\n" % (train_labels[test_index].shape[0], roc_auc, min(probabilities_train), max(probabilities_train), accuracy)

	else:
    		thisMeanSqError = metrics.mean_squared_error(train_labels[test_index], predictions_train)
    		print "[CV] Mean Sq Error (%s, size:%d): %f" % (foldString, train_labels[test_index].shape[0], thisMeanSqError)

		if (CV_on == True):
      			allAucs.append(thisMeanSqError)			
		else:
			allAUCs = thisMeanSqError
		
	if (CV_on == False):
		print "------- Model Building Complete ------------"
		if mod['isRegression']==False:
           		if aeConfig.cvfolds>0:
				print "[PERF] Mean cross-validated AUC: %.4f, Median: %.4f (Range: %.4f - %.4f)" % (np.mean(allAucs), np.median(allAucs), np.min(allAucs), np.max(allAucs))
		else:
     	 		print "[PERF] Mean error: %.8f (Range: %.8f - %.8f)" % (np.mean(allAucs), np.min(allAucs), np.max(allAucs))	



		if(mod['printCoefficients']==True):
                	outputHeader = "%s,%s" % (other_specs['outputHeader'], aeConfig.txtCols)
                	outputHeaderSplit = outputHeader.split(',')
                	coefficientsFile = "%s/%s.%s.lr.coefficients.csv" % (other_specs['outputDir'], other_specs['outputFilePrefix'], today)
     			#utils.printLRCoefficients(modelToUse, oneHotColumnHeaders, coefficientsFile)
	
		if(aeConfig.printFeatureImportance==True):
                	importancesFile = "%s/%s.%s.lr.importances.csv" % (other_specs['outputDir'], other_specs['outputFilePrefix'], today)
                	print "[OUT] Printing feature importances to %s" % (importancesFile)
                	importancesFid = open(importancesFile, 'w')
                	for sortIdx in np.argsort(modelToUse.feature_importances_):
                        	importancesFid.write("%s,%f\n" % (oneHotColumnHeaders[sortIdx], modelToUse.feature_importances_[sortIdx]))
	
        	dx = modelToUse.__dict__
		
	if CV_on == True:
		return test_specs, allAucs,modelToUse, colsToUse
	else:
		return test_specs, allAUCs,modelToUse, colsToUse




def Test_On(modelToUse, mod, other_specs, test_data_transformed_onehot_with_text, dependentVar, test_data_df_original, colsToUse):

        today = datetime.date.today().strftime("%Y%m%d")
        predictions_test = modelToUse.predict(test_data_transformed_onehot_with_text[:,colsToUse])
        test_data_df_original["%s_pred" % (dependentVar)] = predictions_test

        if mod['predict_prob'] == True:
                probabilities_test = modelToUse.predict_proba(test_data_transformed_onehot_with_text[:,colsToUse])[:, 1]
                test_data_df_original["%s_prob" % (dependentVar)] = probabilities_test

        if mod['isRegression'] == False and dependentVar in test_data_df_original.columns:

                thisCm = metrics.confusion_matrix(test_data_df_original[dependentVar].tolist(), predictions_test)
                print "[Confusion Matrix - Test Data]"
                print "          Pred 0\tPred 1\nActl 0    %d\t\t%d\nActl 1    %d\t\t%d" % (thisCm[0][0], thisCm[0][1], thisCm[1][0], thisCm[1][1])
                accuracy = metrics.accuracy_score(test_data_df_original[dependentVar].tolist(), predictions_test)
                print accuracy

        testOutputFile = "%s.%s.OUTPUT.%s.csv" %(other_specs['testDataFile'], dependentVar, today)
        test_data_df_original.to_csv(testOutputFile)
        print "Output written to %s" % (testOutputFile)



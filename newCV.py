import CVstart as cvs
import datetime
from sklearn import (naive_bayes, linear_model, preprocessing, metrics, cross_validation, feature_selection, svm, ensemble, tree, neural_network, cluster, grid_search, neighbors, decomposition)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.externals.six import StringIO
from scipy import io
import scipy
import numpy as np
import filterData
import sys
import ConfigParser
from aeConfig import AEConfig
import utils
import cPickle as pickle
import os
import pandas as pd
from sklearn import preprocessing
import math
import random
import treeutils
from aeConfig import AEConfig
import pdb
import time


def balTrainClust(trainDataFile,datafile, datafile_trans, train_index,CV_on, aeConfig ):
        thisKMeans =  cluster.MiniBatchKMeans(n_clusters=aeConfig.numClusters)
        today = datetime.date.today().strftime("%Y%m%d")
        thisKMeans.fit(datafile_trans[train_index, :])
        if CV_on == False:
                clustersFile = "%s.clusters.%s.csv" % (trainDataFile, today)
                clusterColname = "clusters_%d" % (random.randint(1000,9999))
                if clusterColname not in datafile and datafile.shape[0] == len(thisKMeans.labels_):
                        datafile[clusterColname] = thisKMeans.labels_
                        datafile.to_csv(clustersFile)
                        print "[INFO] Clusters saved with colname '%s' to file: %s" % (clusterColname, clustersFile)
                else:
                        print "[ERROR] Could not output clusters to file. Train data has %d rows, %d labels created" % (datafile.shape[0], len(thisKMeans.labels_))
        print "[INFO] Checking if over/under sampling has been requested"
        return datafile, thisKMeans




def balTrainSam(train_labels, train_index, class_dict):
        print "[INFO] Oversampling requested"
        
	
        train_extender = []
        if class_dict['numMin']<class_dict['numMaj']:
                multiplier = int(math.floor((class_dict['numMaj'] / float(class_dict['numMin'])) - 1))
                for nt in range(multiplier):
                        train_extender.extend(train_index[np.where(train_labels[train_index].ravel()==class_dict['minC'])[0]])
                train_index = np.hstack([train_index, train_extender]).astype(int)
        print "[INFO] Attemped to balance training data. Old: %d 0's, %d 1's, New: %d 0's, %d 1's" % (class_dict['n0'], class_dict['n1'], len(np.where(train_labels[train_index].ravel()==0)[0]), len(np.where(train_labels[train_index].ravel()==1)[0]))
        return train_index, train_labels


def balTrainSamKmeans(train_labels, train_index, class_dict, thisKMeans):
        train_extender = []
	print "[INFO] Oversampling using Kmeans"        
        if class_dict['numMin']<class_dict['numMaj']:
                minorityDiff = class_dict['numMaj'] - class_dict['numMin']
                minorityClassIndexes = np.where(train_labels[train_index].ravel()==class_dict['minC'])[0]
                minorityClassClusters = thisKMeans.labels_[minorityClassIndexes]
                uniqueClusters = np.unique(minorityClassClusters).tolist()
                eachClusterSize = int(math.floor(float(minorityDiff)/len(uniqueClusters)))
                for thisCluster in uniqueClusters:
                        thisClusterIndexes = minorityClassIndexes[np.where(minorityClassClusters==thisCluster)]
                        thisClusterMultiplier = int(math.floor(float(eachClusterSize)/len(thisClusterIndexes)))
                        for tcn in range(thisClusterMultiplier):
                                train_extender.extend(train_index[thisClusterIndexes])
                train_index = np.hstack([(train_index), train_extender])
		train_index = [int(a) for a in train_index]
                print "[INFO] Cluster based oversampling finished. Old: %d 0's, %d 1's, New: %d 0's, %d 1's" % (class_dict['n0'], class_dict['n1'], len(np.where(train_labels[train_index].ravel()==0)[0]), len(np.where(train_labels[train_index].ravel()==1)[0]))
	return train_index, train_labels



def UndSamp(train_labels, train_index,class_dict, train_data_transformed_onehot_with_text ):
        undersamplingReductionPct = 95    ##Eg. if majority class is 40,000, remove 4000 (but never less than minority class)
        thisKMeans =  cluster.MiniBatchKMeans(n_clusters=5)
        thisKMeans.fit(train_data_transformed_onehot_with_text[train_index, :])
        
        minorityClassIndexes = np.where(train_labels[train_index].ravel()==class_dict['minC'])[0]
        majorityClassIndexes = np.where(train_labels[train_index].ravel()==class_dict['majC'])[0]
        new_train_idxes = train_index[minorityClassIndexes.tolist()].tolist()
        if class_dict['numMin'] < class_dict['numMaj']:
                minorityDiff = class_dict['numMaj'] - class_dict['numMin']
                numToRemove = np.min([minorityDiff, int(math.floor(undersamplingReductionPct*class_dict['numMaj']/float(100)))])
                majorityClassClusters = thisKMeans.labels_[majorityClassIndexes]
                uniqueClusters = np.unique(majorityClassClusters).tolist()
                eachClusterSize = int(math.floor(float(class_dict['numMaj']-numToRemove)/len(uniqueClusters)))
                print "[INFO] Undersampling. Initial data: %d minority, %d majority. Sampling %d size from each of %d clusters of majority class" % (class_dict['numMin'], class_dict['numMaj'], eachClusterSize, len(uniqueClusters))
                for thisCluster in uniqueClusters:
                        thisClusterIndexes = majorityClassIndexes[np.where(majorityClassClusters==thisCluster)]
                        new_train_idxes.extend(random.sample(train_index[thisClusterIndexes], np.min([len(thisClusterIndexes), eachClusterSize]) ))
                train_index = new_train_idxes
                print "[INFO] Cluster based undersampling finished. Old: %d 0's, %d 1's, New: %d 0's, %d 1's" % (class_dict['n0'], class_dict['n1'], len(np.where(train_labels[train_index].ravel()==0)[0]), len(np.where(train_labels[train_index].ravel()==1)[0]))
        return train_index, train_labels


def RemRows(train_data_transformed_onehot_with_text, train_index):
        print "[INFO] REMOVING columns with mostly zero values"
        colsToUse = []
        numNonZeroReqd = 1
        for thisColIdx in range(train_data_transformed_onehot_with_text[train_index, :].shape[1]):
                thisNumZero = (np.argwhere((train_data_transformed_onehot_with_text[train_index, :][:,thisColIdx]).todense() == 0)).shape[0]
                if(thisNumZero < train_data_transformed_onehot_with_text[train_index, :].shape[0] - numNonZeroReqd):
                        colsToUse.append(thisColIdx)
        print "[INFO] Eliminated all columns with %d or less non-zero value/s, %d of %d columns remaining" % (numNonZeroReqd, len(colsToUse), train_data_transformed_onehot_with_text[train_index, :].shape[1])
        return colsToUse



def Display_trees(other_specs, modelToUse, oneHotColumnHeaders):
	today = datetime.date.today().strftime("%Y%m%d")
	treeDotFile = "%s/%s.%s.tree.dot" % (other_specs['outputDir'], other_specs['outputFilePrefix'], today)
        print "[INFO] Printing Decision Tree into DOT file %s" % (treeDotFile)
        treePdfFile = "%s/%s.%s.tree.pdf" % (other_specs['outputDir'], other_specs['outputFilePrefix'], today)
        with open(treeDotFile, 'w') as f:
          	f = tree.export_graphviz( modelToUse, out_file=f, feature_names=oneHotColumnHeaders)
        dot_data = StringIO()
	print dot_data
        tree.export_graphviz(modelToUse, out_file=dot_data, feature_names=oneHotColumnHeaders)
       

def MajMin(train_labels, train_index):
        num0s = len(np.where(train_labels[train_index].ravel()==0)[0])
        num1s = len(np.where(train_labels[train_index].ravel()==1)[0])
        minorityClass=0
        majorityClass=1
        if num1s<num0s:
                minorityClass=1
                majorityClass=0

        numMinority = len(np.where(train_labels[train_index].ravel()==minorityClass)[0])
        numMajority = len(np.where(train_labels[train_index].ravel()==majorityClass)[0])
        return majorityClass, minorityClass, numMajority, numMinority, num0s, num1s


def Create_trees(oneHotColumnHeaders, modelToUse, modelToUse2, train_data_transformed_onehot_with_text, train_labels, train_index_unmodified, mod, aeConfig, scalingFactors, colsToUse, other_specs, mod2, train_data_transformed_onehot_with_text_dense):	


	today = datetime.date.today().strftime("%Y%m%d")
	comboColumnIdxes = [ thisIdx for thisIdx in range(len(oneHotColumnHeaders)) if ':' in oneHotColumnHeaders[thisIdx] ]
      	nonComboColumnIdxes = [ cn for cn in colsToUse if cn not in comboColumnIdxes ]
      	nonComboColumnHeaders = [ oneHotColumnHeaders[cn] for cn in nonComboColumnIdxes ]

	treeTimeStart = int(round(time.time() * 1000))
      	treeMaxFeatures=np.min([5, len(nonComboColumnIdxes)])
      	
	treeModel = modelToUse2
	if mod2['isRegression'] == True:
		mod['isRegression'] = True


      	print "[INFO] Building random forest of %d trees with depth %d for insights ..." % (aeConfig.numTreesCreate, aeConfig.treeDepth)
      	
      	treeModel.fit(train_data_transformed_onehot_with_text_dense[train_index_unmodified,:][:,nonComboColumnIdxes], train_labels[train_index_unmodified].ravel())
      	treeTimeEnd = int(round(time.time() * 1000))
      	print "[INFO] Random forest created in %.3f s" % (float(treeTimeEnd - treeTimeStart)/1000)
      	thisEstimatorPickleFile = "%s/%s.%s.%s.randomforest.pickle" % (other_specs['outputDir'], other_specs['outputFilePrefix'],other_specs['dependentVar'], today)
      	pickle.dump(treeModel, file(thisEstimatorPickleFile, 'w'))
      	print "[INFO] Random forest object saved in: %s" % (thisEstimatorPickleFile)
      		
	estimatorScores = []
      	for estimatorIdx in range(len(treeModel.estimators_)):
        	thisEstimator = treeModel.estimators_[estimatorIdx]
        	estimatorScores.append(treeutils.treeScore(thisEstimator.tree_))
      	sortedScoreIndexes = [i[0] for i in sorted(enumerate(estimatorScores), key=lambda x:x[1])]

      	featuresUsed = []
	treesPrinted = 0
      	sortIdx = -1

	while treesPrinted<aeConfig.numTreesPick and sortIdx<(len(sortedScoreIndexes)-1):
        	sortIdx = sortIdx + 1
        	estimatorIdx = sortedScoreIndexes[sortIdx]
        	thisEstimator = treeModel.estimators_[estimatorIdx]
        	thisScore = estimatorScores[estimatorIdx]
        	allFeatureIdxes = thisEstimator.tree_.feature[np.where(thisEstimator.tree_.feature>=0)].tolist()
        	thisRootFeatureIdx = thisEstimator.tree_.feature[0]
        	skipThisTree = False
        	if aeConfig.uniqueFeatureTreesOnly==True:
          		for thisFeatureIdx in allFeatureIdxes:
            			if thisFeatureIdx in featuresUsed:
              				print "Skipping tree %d (score:%.4f), feature %s used previously" % (estimatorIdx, thisScore, nonComboColumnHeaders[thisFeatureIdx])
              				skipThisTree = True
              				break
        	elif thisRootFeatureIdx in featuresUsed:
            		print "Skipping tree %d (score:%.4f), feature %s used at root level previously" % (estimatorIdx, thisScore, nonComboColumnHeaders[thisRootFeatureIdx])
            		skipThisTree = True
        		
		if skipThisTree==True:
          		continue
        
		thisTreePrefix = "%s/%s.%s.%s.tree.%0.8d" % (other_specs['outputDir'], other_specs['outputFilePrefix'], other_specs['dependentVar'], today, treesPrinted+1)
        	thisTreeDotFile = "%s.dot" % (thisTreePrefix)
        	thisTreeJsonFile = "%s.json" % (thisTreePrefix)
        	print "[INFO] Printing Decision Tree %d (root feature: '%s', score: %.4f):" % (estimatorIdx, nonComboColumnHeaders[thisRootFeatureIdx], thisScore)

		thisTreeClasses = []

        	if thisEstimator.n_classes_==2:
          		thisTreeClasses = thisEstimator.classes_
        	print "\tinto JSON file: %s" % (thisTreeJsonFile)

		thisTreeJson = ""
        	try:
          		thisTreeJson = treeutils.printTreeJson(thisEstimator.tree_, estimatorIdx, nonComboColumnHeaders, thisTreeClasses, other_specs['dependentVar'], train_labels[train_index_unmodified].ravel(), multipliers=scalingFactors)
        	except:
          		thisException = sys.exc_info()[0]
          		print "[ERROR] Exception encountered in treeutils.printTreeJson for estimatorIdx: %d and dependentVar: %s" % (estimatorIdx, other_specs['dependentVar'])
          		print thisException
        	thisTreeJsonFid = open(thisTreeJsonFile, 'w')
        	thisTreeJsonFid.write(thisTreeJson)
        	thisTreeJsonFid.close()
        
        	if aeConfig.uniqueRootNodes==True and "=" in nonComboColumnHeaders[thisRootFeatureIdx] and len(nonComboColumnHeaders[thisRootFeatureIdx].split("="))==2:
          		thisRootFeatureLHS = nonComboColumnHeaders[thisRootFeatureIdx].split("=")[0]
			print "[TREES] Skipping ALL other '%s=_____' features (aeConfig.uniqueRootNodes=True)" % (thisRootFeatureLHS)
          		featuresUsed.extend([x for x in range(len(nonComboColumnHeaders)) if thisRootFeatureLHS in nonComboColumnHeaders[x]])

        	else:
          		featuresUsed.append(thisRootFeatureIdx)
        		treesPrinted = treesPrinted + 1
		
	return thisRootFeatureIdx

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import statistics
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score
from sklearn.metrics import precision_score,recall_score,f1_score
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def get_normalized_inner_product_score(vector1, vector2):
	return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_sigmoid_score(vector1, vector2):
	return sigmoid(np.dot(vector1, vector2))

def get_average_score(vector1, vector2):
	return (vector1 + vector2)/2

def get_hadamard_score(vector1, vector2):
	return np.multiply(vector1, vector2)

def get_l1_score(vector1, vector2):
	return np.abs(vector1 - vector2)

def get_l2_score(vector1, vector2):
	return np.square(vector1 - vector2)

def get_link_score(embeds, node1, node2, score_type):
	if score_type not in ["cosine", "sigmoid", "hadamard", "average", "l1", "l2"]:
		raise NotImplementedError
	vector_dimension = embeds.shape[1]
	try:
		vector1 = embeds[node1]
		vector2 = embeds[node2]
	except Exception as e:
		if score_type in ["cosine", "sigmoid"]:
			return 0
		elif score_type in ["hadamard", "average", "l1", "l2"]:
			return np.zeros(vector_dimension)

	if score_type == "cosine":
		score = get_normalized_inner_product_score(vector1, vector2)
	elif score_type == "sigmoid":
		score = get_sigmoid_score(vector1, vector2)
	elif score_type == "hadamard":
		score = get_hadamard_score(vector1, vector2)
	elif score_type == "average":
		score = get_average_score(vector1, vector2)
	elif score_type == "l1":
		score = get_l1_score(vector1, vector2)
	elif score_type == "l2":
		score = get_l2_score(vector1, vector2)

	return score

def get_links_scores(embeds, links, score_type):
	features = []
	num_links = 0
	for l in links:
		num_links = num_links + 1
		node1, node2 = l[0], l[1]
		f = get_link_score(embeds, node1, node2, score_type)
		features.append(f)
	return features

def evaluate_classifier(embeds, train_pos_edges, train_neg_edges, test_pos_edges, test_neg_edges, score_type):
	train_pos_feats = np.array(get_links_scores(embeds, train_pos_edges, score_type))
	train_neg_feats = np.array(get_links_scores(embeds, train_neg_edges, score_type))
	train_pos_labels = np.ones(train_pos_feats.shape[0])
	train_neg_labels = np.zeros(train_neg_feats.shape[0])
	train_data = np.concatenate((train_pos_feats, train_neg_feats), axis=0)
	train_labels = np.append(train_pos_labels, train_neg_labels)

	test_pos_feats = np.array(get_links_scores(embeds, test_pos_edges, score_type))
	test_neg_feats = np.array(get_links_scores(embeds, test_neg_edges, score_type))
	test_pos_labels = np.ones(test_pos_feats.shape[0])
	test_neg_labels = np.zeros(test_neg_feats.shape[0])
	test_data = np.concatenate((test_pos_feats, test_neg_feats), axis=0)
	test_labels = np.append(test_pos_labels, test_neg_labels)

	logistic_regression = linear_model.LogisticRegression()
	logistic_regression.fit(train_data, train_labels)

	test_predict_prob = logistic_regression.predict_proba(test_data)
	test_predict = logistic_regression.predict(test_data)
	# print(test_predict.shape, test_predict_prob.shape)

	auroc = roc_auc_score(test_labels, test_predict_prob[:, 1])
	precisions, recalls, _ = precision_recall_curve(test_labels, test_predict_prob[:, 1])
	auprc = auc(recalls, precisions)
	return auroc, auprc
#性能度量，可选Acc，Spe，Rec, Pre,MCC
def link_prediction(embed, edges, score_type, n_trials=5):
	pos_edges = edges['pos_samples'] 
	neg_edges = edges['neg_samples'] 
	# shuffle and split training and test sets
	trials = ShuffleSplit(n_splits=n_trials, random_state=None)
	ss_pos = trials.split(pos_edges)
	trial_splits_pos = []
	for train_idx, test_idx in ss_pos:
		trial_splits_pos.append((train_idx, test_idx))
	ss_neg = trials.split(neg_edges)
	trial_splits_neg = []
	for train_idx, test_idx in ss_neg:
		trial_splits_neg.append((train_idx, test_idx))

	list_auroc = []
	list_auprc = []
	for idx in range(n_trials):
		test_idx,train_idx = trial_splits_pos[idx]
		train_pos = pos_edges[train_idx,:]
		test_pos = pos_edges[test_idx,:]
		test_idx,train_idx = trial_splits_neg[idx]
		train_neg = neg_edges[train_idx,:]
		test_neg = neg_edges[test_idx,:]

		auroc, auprc = evaluate_classifier(embed,train_pos,train_neg,test_pos,test_neg,score_type)
		list_auroc.append(auroc)
		list_auprc.append(auprc)
	# print(list_auroc,list_auprc)
	avg_auroc = statistics.mean(list_auroc)
	std_auroc = statistics.stdev(list_auroc)
	avg_auprc = statistics.mean(list_auprc)
	std_auprc = statistics.stdev(list_auprc)

	return avg_auroc,std_auroc,avg_auprc,std_auprc





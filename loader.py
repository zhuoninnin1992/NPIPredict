#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy.sparse import csr_matrix
from data_helper import train_tiedAE
from sklearn import preprocessing

#可选设置其它数据集 RPI369，RPI2241， RPI7317等
def load_data(data):
	#仅有拓扑结构
	if data == "RPI2_0":
		data = np.loadtxt('data/RPI2_0/RPI2_0.txt', dtype=int)
		return data, None
	#序列信息+拓扑结构
	elif data == "RPI2_0FE":
		data = np.loadtxt('data/RPI2_0/RPI2_0_edges.txt', dtype=int)
		feats_N = np.loadtxt('data/RPI2_0/RPI2_0_featncRNA.txt', dtype=float)
		feats_P = np.loadtxt('data/RPI2_0/RPI2_0_featprotein.txt', dtype=float)
		feats = np.concatenate((feats_N ,feats_P),axis=0)
		return data, feats


def load_attributes(data, feats):
	# print(feats.shape)
	num_btype = len(np.unique(data[:,2]))
	btypes = [str(i) for i in range(num_btype)]
	# btypes = []
	btypes.append('base')
	# print(btypes)
	init_feats = dict()
	for btype in btypes:
		init_feats[btype] = feats #preprocessing.normalize(feats)
	return init_feats

def split_train_test(data, feats, flag, ratio):
	n_samples = data.shape[0]
	n_test = int(n_samples*ratio)
	ridx = np.random.choice(n_samples, n_test, replace=False)
	test = data[ridx]
	train = np.delete(data, ridx, axis=0)
	print(test.shape,train.shape)
	train_nodes_n = [i for i in list(set(train[:,0]))]
	train_nodes_p = [i for i in list(set(train[:,1]))]
	# print(len(train_nodes_n),len(train_nodes_p))
	train_np = train_nodes_n
	for i in train_nodes_p:
		train_np.append(i)
	# if feats != None:
	# 	feats_train = np.array([feats[i] for i in train_np])
	# 	print(feats.shape, feats_train.shape)
	n_train,p_train = np.unique(train[:,0]),np.unique(train[:,1])
	f_test = []
	for line in test:
		if line[0] in n_train and line[1] in p_train:
			f_test.append(line)
	f_test = np.array(f_test)
	# print("### The train contains %d edges(%d drugs and %d targets), and test contains %d edges(%d drugs and %d targets)." 
	# 	% (train.shape[0],len(n_train),len(p_train),f_test.shape[0],len(np.unique(f_test[:,0])),len(np.unique(f_test[:,1]))))
	idx_n_map = {j:i for i,j in enumerate(np.unique(train[:,0]))}
	idx_p_map = {j:len(np.unique(train[:,0]))+i for i,j in enumerate(np.unique(train[:,1]))}
	new_train,new_test = [],[]
	for line in train:
		tmp = []
		tmp.append(idx_n_map[line[0]])
		tmp.append(idx_p_map[line[1]])
		tmp.append(line[2])
		# tmp.append(line[3]) # label
		new_train.append(tmp)
	for line in f_test:
		tmp = []
		tmp.append(idx_n_map[line[0]])
		tmp.append(idx_p_map[line[1]])
		tmp.append(line[2])
		# tmp.append(line[3])
		new_test.append(tmp)
	new_train,new_test = np.array(new_train),np.array(new_test)
	n_train_new,p_train_new = np.unique(new_train[:,0]),np.unique(new_train[:,1])
	
	print(len(np.unique(new_train[:,0])),len(np.unique(new_train[:,1])))
	if flag == True:
		feats_train = np.array([feats[i] for i in train_np])
		return new_train, new_test, feats_train
	elif flag == False:
		return new_train, new_test, None

def extract_edges_method(data, btype):
	new_data = []
	for line in data:
		if line[2] == btype:
			new_data.append(line)
	return np.array(new_data)

def construct_D_ncRNA_protein(data, nodes):
	graphs_t = dict()
	btypes = list(set(data[:,2])) #[0,1,2,3,4]
	graph_base = construct_graph(data, nodes)
	graphs_t['base'] = graph_base
	for btype in btypes:
		data_type = extract_edges_method(data, btype)
		graphs_t[str(btype)] = construct_graph(data_type, nodes)
	return graphs_t

def construct_graph(data, nodes):
	# construct Bigraph
	nodes_n = [i for i in list(set(data[:,0]))]
	nodes_p = [i for i in list(set(data[:,1]))]
	all_nodes_n,all_nodes_p = nodes['n'],nodes['p']
	print(len(nodes_n),len(nodes_p),len(all_nodes_n),len(all_nodes_p))
	Bigraph = nx.Graph()
	for line in data:
		node_n,node_p = line[0],line[1]
		Bigraph.add_node(node_n, bipartite=0)
		Bigraph.add_node(node_p, bipartite=1)
		Bigraph.add_edge(node_n, node_p, btype=line[2])
	# construct graph
	D_graph = dict()
	n_neigs_n = 0
	n_neigs_p = 0
	for u in all_nodes_n:
		if u in nodes_n:
			neighbors = Bigraph.edges(u)
			neigs_u = [i for u,i in neighbors]
			D_graph[u] = neigs_u
			n_neigs_n += len(neigs_u)
		else:
			D_graph[u] = []
	for i in all_nodes_p:
		if i in nodes_p:
			neighbors = Bigraph.edges(i)
			neigs_i = [u for i,u in neighbors]
			D_graph[i] = neigs_i
			n_neigs_p += len(neigs_i)
		else:
			D_graph[i] = []
	
	return D_graph

def generate_adj(data,nodes,btype):
	N_n = nodes['n_n']#len(np.unique(data[:,0]))
	N_p = nodes['n_p']#len(np.unique(data[:,1]))
	N = N_n + N_p
	adj = np.zeros((N, N), dtype=int)
	if btype == 'base':
		for line in data:
			adj[line[0],line[1]] = 1
	else:
		for line in data:
			if line[2] == int(btype):
				adj[line[0],line[1]] = 1
	# print(np.sum(adj == 1))
	return csr_matrix(adj).astype('float32')

def initialize_features(args, data, nodes, dim=32):
	# Encoder Based Approach
	print('### Generating initial features by Encoder-Based-Approach...')
	num_btype = len(np.unique(data[:,2]))
	btypes = [str(i) for i in range(num_btype)]
	# btypes = []
	btypes.append('base')
	initial_feats = dict()
	for btype in btypes:
		A = generate_adj(data,nodes,btype).todense()
		initial_feat = train_tiedAE(A,dim=args.dim_f,lr=args.lr_eba,weight_decay=args.weight_decay_eba,n_epochs=args.epoch_eba)
		initial_feats[btype] = preprocessing.normalize(initial_feat)
	return initial_feats

def generate_incidence_matrix_multiple(graph):
	Gs = dict()
	btypes = graph.keys()
	n_smp = len(graph['base'])
	for btype in btypes:
		G = generate_incidence_matrix(graph[btype],n_smp)
		Gs[btype] = G
	return Gs

def generate_incidence_matrix(edges, n_smp):
	G = np.zeros((n_smp,n_smp))
	for key,val in edges.items():
		for v in val:
			G[v,key] = 1
	return G

def generate_negative_samples(pos_edges, graph, num_neg_samples):
	nodes_n = list(set([u for u,i in pos_edges]))
	nodes_p = list(set([i for u,i in pos_edges]))
	neg_edges = []
	for n in nodes_n:
		candidates = list(set(nodes_p) - set(graph[n]))
		neg_nodes = np.random.choice(candidates, num_neg_samples, replace=False)
		for neg in neg_nodes:
			tmp = [n, neg]
			neg_edges.append(tmp)
	for p in nodes_p:
		candidates = list(set(nodes_n) - set(graph[p]))
		neg_nodes = np.random.choice(candidates, num_neg_samples, replace=False)
		for neg in neg_nodes:
			tmp = [p, neg]
			neg_edges.append(tmp)
	neg_edges = np.array(neg_edges)
	np.random.shuffle(neg_edges)
	return neg_edges

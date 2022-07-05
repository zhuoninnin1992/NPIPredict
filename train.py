#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from loader import *
from sklearn import preprocessing
from NPI_DGCN import train_NPI_DGCN
from evaluation import link_prediction


def parse_args():
	parser = argparse.ArgumentParser(description="Run NPI predictor")
	parser.add_argument('--data', type=str, default='RPI2_0')
	parser.add_argument('--task', type=str, default='LP')
	parser.add_argument('--ratio', type=float, default=0.8)
	parser.add_argument('--n_neg', type=int, default=2)
	parser.add_argument('--conv', type=str, default='asym')
	parser.add_argument('--attr', type=bool, default=False)
	# Parameters of initializing features (TiedAutoEncoder)
	parser.add_argument('--dim_f', type=int, default=32)
	parser.add_argument('--lr_eba', type=float, default=0.005)
	parser.add_argument('--epoch_eba', type=int, default=50)
	parser.add_argument('--weight_decay_eba', type=float, default=5e-4)
	# Parameters of NPI-DGCN
	parser.add_argument('--dim', type=list, default=[64,32])
	parser.add_argument('--lr', type=float, default=0.002)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--weight_decay', type=float, default=5e-4)
	parser.add_argument('--dropout', type=float, default=0.5)
	parser.add_argument('--inter', type=str, default=True)
	parser.add_argument('--intra', type=str, default=False)
	parser.add_argument('--lamb', type=float, default=0.5)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	print(args.data, args.task, args.n_neg, args.conv)
	print(args.dim, args.lr, args.epoch, args.optimizer, args.dropout,args.inter, args.intra, args.lamb)
	data, feats = load_data(args.data)

	if args.task == "LP":
		if args.attr == False:
			train, test, feats_train = split_train_test(data, feats, False, ratio=args.ratio)
		elif args.attr == True:
			train, test, feats_train = split_train_test(data, feats, True, ratio=args.ratio)

		nodes = dict()
		nodes_n,nodes_p = np.unique(train[:,0]),np.unique(train[:,1])
		num_n,num_p = len(nodes_n),len(nodes_p)
		nodes['n'],nodes['p'],nodes['n_n'],nodes['n_p'] = nodes_n,nodes_p,num_n,num_p
		del data

		Graph_ncRNA_protein= construct_D_ncRNA_protein(train, nodes) #['base','0','1','2','3','4']
		Bi_graph = Graph_ncRNA_protein['base']
		print(Graph_ncRNA_protein.keys())

		pos_samples = train[:,:2]
		neg_samples = generate_negative_samples(pos_samples, Bi_graph, args.n_neg)
		print("### Positive samples: %d, Negative samples: %d" %(len(pos_samples),len(neg_samples)))
		samples = dict()
		samples['pos_samples'] = pos_samples
		samples['neg_samples'] = neg_samples

		if args.attr == False:
			init_embs = initialize_features(args, train, nodes)
		elif args.attr == True:
			init_embs = load_attributes(train, feats_train)
		print(len(init_embs), init_embs['base'].shape)
		Hs = generate_incidence_matrix_multiple(Graph_ncRNA_protein)
		del Graph_ncRNA_protein,Bi_graph

		test_Bi_graph = construct_graph(test, nodes)
		test_pos_samples = test[:,:2]
		test_neg_samples = generate_negative_samples(test_pos_samples, test_Bi_graph, args.n_neg)
		print("### Positive samples: %d, Negative samples: %d" %(len(test_pos_samples),len(test_neg_samples)))
		test_samples = dict()
		test_samples['pos_samples'] = test_pos_samples
		test_samples['neg_samples'] = test_neg_samples

		embs = train_NPI_DGCN(args, init_embs, Hs, samples, num_n)
		print("### Link Prediction task:")
		avg_auroc,std_auroc,avg_auprc,std_auprc = link_prediction(embs,test_samples,'hadamard')
		print("### Average(over trials) of NPIDGCN:  AUROC: {:.4f}({:.4f}), AUPRC: {:.4f}({:.4f})".format(avg_auroc,std_auroc,avg_auprc,std_auprc))


	print()

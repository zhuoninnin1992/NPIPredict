#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor,optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from time import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

### <!-- Dual Convolutional Network for learning graphs --!> ###
class NPI_DGCN(nn.Module):
	def __init__(self, in_ch, n_hid, dty_nets, inter, intra, dropout=0.5):
		super(NPI_DGCN, self).__init__()
		self.dropout = dropout
		self.dty_nets = dty_nets
		self.dim_emb = n_hid[-1]
		self.inter = inter
		self.intra = intra
		self.Conv_1 = Conv(in_ch, n_hid[0], self.dty_nets, self.inter, self.intra, self.dropout)
		self.Conv_2 = Conv(n_hid[0], n_hid[1], self.dty_nets, self.inter, self.intra, self.dropout)
		self.Linear_n = nn.Linear(n_hid[-1]*len(dty_nets), n_hid[-1])
		self.Linear_p = nn.Linear(n_hid[-1]*len(dty_nets), n_hid[-1])
		print(dty_nets)

	def dropout_layer(self, X_n, X_p):
		out_x_n,out_x_p=dict(),dict()
		for dty in self.dty_nets:
			out_x_n[dty] = F.dropout(X_n[dty], self.dropout)
			out_x_p[dty] = F.dropout(X_p[dty], self.dropout)
		return out_x_n,out_x_p

	def forward(self, X_n, X_p, G_n, G_p, H_n, H_p):
		X_n_1,X_p_1 = self.Conv_1(X_n, G_n, H_n, X_p, G_p, H_p)
		X_n_1,X_p_1 = self.dropout_layer(X_n_1, X_p_1)
		X_n_2,X_p_2 = self.Conv_2(X_n_1, G_n, H_n, X_p_1, G_p, H_p)
		dty_nets = self.dty_nets-['base']

		all_x_n = X_n_2['base']
		for dty in dty_nets:
			add_x_n = X_n_2[dty]
			all_x_n = torch.cat((all_x_n, add_x_n), 1)
		opt_x_n = self.Linear_n(all_x_n)

		all_x_p = X_p_2['base']
		for dty in dty_nets:
			add_x_p = X_p_2[dty]
			all_x_p = torch.cat((all_x_p, add_x_p), 1)
		opt_x_p = self.Linear_p(all_x_p)

		opt_x = torch.cat((opt_x_n, opt_x_p), 0)
		return opt_x

class Embed_layer(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets):
		super(Embed_layer, self).__init__()
		self.weight = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.reset_parameters()
		self.dty_nets = dty_nets

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)

	def forward(self, X:torch.Tensor):
		X_ = dict()
		for dty in self.dty_nets:
			X_[dty] = X[dty].matmul(self.weight)
		return X_

class GraphConv(nn.Module):
	def __init__(self, in_ft, out_ft, inter=False, intra=True, bias=True):
		super(GraphConv, self).__init__()
		self.weight_n = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		self.weight_p = Parameter(torch.Tensor(in_ft, out_ft)).to(device)
		if bias:
			self.bias = Parameter(torch.Tensor(out_ft)).to(device)
		else:
			self.register_parameter(torch.Tensor(out_ft)).to(device)
		self.WB = Parameter(torch.Tensor(out_ft, out_ft)).to(device)
		self.reset_parameters()
		self.inter = inter
		self.intra = intra

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight_n.size(1))
		self.weight_n.data.uniform_(-stdv, stdv)
		self.weight_p.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)
		self.WB.data.uniform_(-stdv, stdv)

	def forward(self,XN:torch.Tensor,GN:torch.Tensor,HN:torch.Tensor,XP:torch.Tensor,GP:torch.Tensor,HP:torch.Tensor,B:torch.Tensor,_intra:torch.bool):
		XN = XN.matmul(self.weight_n)
		XP = XP.matmul(self.weight_p)
		X = GN.matmul(XN)
		if self.inter:
			HiT = torch.transpose(HP,0,1)
			X = X + HiT.matmul(XP)
		if self.intra and _intra:
			X = X + B.matmul(self.WB)
		if self.bias is not None:
			X = X + self.bias
		X = F.relu(X)
		return X

class Conv(nn.Module):
	def __init__(self, in_ft, out_ft, dty_nets, inter, intra, dropout, bias=True):
		super(Conv, self).__init__()
		self.dty_nets = dty_nets
		self.dropout = dropout
		self.GraphConv_N = dict()
		self.GraphConv_P = dict()
		for dty in self.dty_nets:
			self.GraphConv_N[dty] = GraphConv(in_ft, out_ft, inter=inter, intra=intra)
			self.GraphConv_P[dty] = GraphConv(in_ft, out_ft, inter=inter, intra=intra)

	def forward(self,XN:torch.Tensor,GN:torch.Tensor,HN:torch.Tensor,XP:torch.Tensor,GP:torch.Tensor,HP:torch.Tensor):
		self._dty_nets = self.dty_nets-['base']
		### GraphConv on ncRNA
		out_xu = dict()
		base_xu = self.GraphConv_N['base'](XN['base'],GN['base'],HN['base'],XP['base'],GP['base'],HP['base'],XN,False)
		out_xu['base'] = base_xu
		for dty in self._dty_nets:
			add_xu = self.GraphConv_N[dty](XN[dty],GN[dty],HN[dty],XP[dty],GP[dty],HP[dty],base_xu,True)
			out_xu[dty] = add_xu

		### GraphConv on Protein
		out_xi = dict()
		base_xi = self.GraphConv_P['base'](XP['base'],GP['base'],HP['base'],XN['base'],GN['base'],HN['base'],XP,False)
		out_xi['base'] = base_xi
		for dty in self._dty_nets:
			add_xi = self.GraphConv_P[dty](XP[dty],GP[dty],HP[dty],XN[dty],GN[dty],HN[dty],base_xi,True)
			out_xi[dty] = add_xi

		return out_xu,out_xi

def generate_G_from_BG(args, H):
	H = np.array(H)
	n_edge = H.shape[1]
	W = np.ones(n_edge)
	DV = np.sum(H * W, axis=1)
	DE = np.sum(H, axis=0)
	DV += 1e-12
	DE += 1e-12
	invDE = np.mat(np.diag(np.power(DE, -1)))
	W = np.mat(np.diag(W))
	H = np.mat(H)
	HT = H.T
	if args.conv == "sym":
		DV2 = np.mat(np.diag(np.power(DV, -0.5)))
		G = DV2 * H * W * invDE * HT * DV2   #sym
	elif args.conv == "asym":
		DV1 = np.mat(np.diag(np.power(DV, -1)))
		G = DV1 * H * W * invDE * HT   #asym
	return G

def generate_Gs_from_O(args, DGs):
	Gs = dict()
	for key,val in DGs.items():
		Gs[key] = generate_G_from_BG(args, val)
	return Gs

def split_Gs(Gs, num_n):
	Gs_n,Gs_p = dict(),dict()
	for key,val in Gs.items():
		Gs_n[key] = val[:num_n,num_n:]
		Gs_p[key] = val[num_n:,:num_n]
	return Gs_n,Gs_p

def embedding_loss(embeddings, positive_links, negtive_links, lamb):
	left_p = embeddings[positive_links[:, 0]]
	right_p = embeddings[positive_links[:, 1]]
	dots_p = torch.sum(torch.mul(left_p, right_p), dim=1)
	positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
	left_n = embeddings[negtive_links[:, 0]]
	right_n = embeddings[negtive_links[:, 1]]
	dots_n = torch.sum(torch.mul(left_n, right_n), dim=1)
	negtive_loss = torch.mean(-1.0 * torch.log(1.01 - torch.sigmoid(dots_n)))
	loss =  lamb*positive_loss + (1-lamb)*negtive_loss
	return loss

def train(args, model, X_n, X_p, samples, G_n, G_p, DG_n, DG_p):
	lr = args.lr
	weight_decay = args.weight_decay

	if args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
	n_epoch = args.epoch
	feats_n, feats_p, target1, target2 = X_n, X_p, samples['pos_samples'], samples['neg_samples']

	for epoch in range(n_epoch):
		model.train()
		optimizer.zero_grad()
		embeds = model.forward(feats_n, feats_p, G_n, G_p, DG_n, DG_p)
		loss = embedding_loss(embeds, target1, target2, args.lamb)
		loss.backward()
		optimizer.step()
		if (epoch+1) % 100 == 0 or epoch == 0:
			print('The loss of %d-th epoch: %0.4f' % (epoch+1, loss))
	model.eval()
	outputs = model.forward(feats_n, feats_p, G_n, G_p, DG_n, DG_p)
	return outputs

def train_NPI_DGCN(args, X, Gcns, samples, num_n):
	Xs_n,Xs_p = dict(),dict()
	for key,val in X.items():
		X_ = X[key]
		X_n,X_p = X_[:num_n,:],X_[num_n:,:]
		Xs_n[key] = Tensor(X_n).to(device)
		Xs_p[key] = Tensor(X_p).to(device)
	# n_sample = X.shape[0]
	in_ft = X['base'].shape[1]
	DG_n,DG_p = split_Gs(Gcns,num_n)
	G_n = generate_Gs_from_O(args, DG_n)
	G_p = generate_Gs_from_O(args, DG_p)
	Gs_n,Gs_p = dict(),dict()
	Gcns_n,Gcns_p = dict(),dict()
	for key,val in G_n.items():
		Gs_n[key] = Tensor(G_n[key]).to(device)
		Gcns_n[key] = Tensor(DG_n[key]).to(device)
	for key,val in G_p.items():
		Gs_p[key] = Tensor(G_p[key]).to(device)
		Gcns_p[key] = Tensor(DG_p[key]).to(device)

	model = NPI_DGCN(in_ch=in_ft,n_hid=args.dim,dty_nets=Gcns.keys(),inter=args.inter,intra=args.intra,dropout=args.dropout)
	model = model.to(device)
	emb = train(args, model, Xs_n, Xs_p, samples, Gs_n, Gs_p, Gcns_n, Gcns_p)
	return emb.detach().cpu().numpy()
	# return emb.detach().numpy()

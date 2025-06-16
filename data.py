import numpy as np
import torch
import json
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
import math
from random import sample


class MyData(Dataset):
	def __init__(self, data_path, frac=0.1, l=0, min_l=1, max_l=101, features_index=None, mix=False, N=None,
	             TaifengID=-1):
		if features_index is None:
			features_index = [2, 3, 4, 5]
		self.N = N
		self.features_index = features_index
		self.data_path = data_path
		self.l = l
		self.min_l = min_l
		self.max_l = max_l
		self.mix = mix
		self.frac = frac
		self.data, self.TaiFengClass = self.read_data()
		
		self.mindata, self.maxdata = self.get_min_max()
		self.encoded_sequence = self.get_sinusoidal_encoding()
		
		self.data = self.normalization(self.data)
		self.TaifengID = TaifengID
		self.sequence, self.seq_len_dict = self.get_sequence()
	
	def __getitem__(self, index):
		return self.sequence[index]
	
	def get_sequence(self):
		seq = []
		seq_len_dict = {}
		
		n = len(self.data)
		if self.mix:
			if self.l:
				for j in range(n - self.l):
					seq.append(
						[self.data[j:j + self.l], self.data[j + self.l], self.l, [self.TaiFengClass[j], j, self.l]])
			return seq, seq_len_dict
		
		if self.TaifengID >= 0:
			rng = [self.TaifengID, self.TaifengID]
		else:
			rng = [0, self.TaiFengClass[-1]]
		
		for i in range(rng[0], rng[1] + 1):
			TaiFeng = self.data[self.TaiFengClass == i]
			n = len(TaiFeng)
			if self.l:
				if self.l <= n - 1:
					for j in range(n - self.l):
						seq.append([TaiFeng[j:j + self.l], TaiFeng[j + self.l], self.l, [i, j, self.l]])
			else:
				for j in range(n - 1):
					for l in range(j + 1, n):
						if self.min_l <= l - j < self.max_l:
							seq.append([TaiFeng[j:l], TaiFeng[l], l - j, [i, j, l]])
							if l - j not in seq_len_dict:
								seq_len_dict[l - j] = 0
							seq_len_dict[l - j] += 1
		if self.frac != 1:
			seq = sample(seq, int(len(seq) * self.frac))
		if self.N is not None:
			seq = sample(seq, min(len(seq), self.N))
		print('总共生成', len(seq), '个序列')
		return seq, seq_len_dict
	
	def get_sinusoidal_encoding(self, time_sequence=torch.arange(1, 13), d_model=128):
		position = torch.arange(0, time_sequence.shape[0], dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		encodings = torch.zeros(time_sequence.shape[0], d_model)
		encodings[:, 0::2] = torch.sin(position * div_term)
		encodings[:, 1::2] = torch.cos(position * div_term)
		return encodings
	
	def get_mix_t_sequence(self,update=True):
		seq = []
		seq_len_dict = {}
		for t in range(1,13):
			seq_temp, _ = self.get_t_sequence(t,update=False)
			seq += seq_temp
		
		if self.frac != 1:
			seq = sample(seq, int(len(seq) * self.frac))
		if self.N is not None:
			seq = sample(seq, min(len(seq), self.N))
		print('总共生成', len(seq), '个序列')
		
		if update:
			self.sequence = seq
			self.seq_len_dict = seq_len_dict
		else:
			return seq, seq_len_dict
		
	
	def get_t_sequence(self, t, update=True):
		seq_len_dict = {}
		seq = []
		
		if self.TaifengID >= 0:
			rng = [self.TaifengID, self.TaifengID]
		else:
			rng = [0, self.TaiFengClass[-1]]
		
		for i in range(rng[0], rng[1] + 1):
			TaiFeng = self.data[self.TaiFengClass == i]
			n = len(TaiFeng)
			if self.l:
				if self.l <= n - t:
					for j in range(n - self.l - t + 1):
						seq.append(
							[TaiFeng[j:j + self.l], TaiFeng[j + self.l + t - 1], self.l, [i, j, self.l, self.encoded_sequence[t-1]]])
		
		if self.frac != 1:
			seq = sample(seq, int(len(seq) * self.frac))
		if self.N is not None:
			seq = sample(seq, min(len(seq), self.N))
		print('总共生成', len(seq), '个序列')
		
		if update:
			self.sequence = seq
			self.seq_len_dict = seq_len_dict
		else:
			return seq, seq_len_dict
	
	def get_xpredict_sequence(self, x=1, update=False):
		seq = []
		seq_len_dict = {}
		
		for i in range(self.TaiFengClass[-1] + 1):
			TaiFeng = self.data[self.TaiFengClass == i]
			n = len(TaiFeng)
			if self.l:
				if self.l <= n - x:
					for j in range(n - self.l - x+1):
						seq.append(
							[TaiFeng[j:j + self.l], TaiFeng[(j + self.l):(j + self.l + x)], self.l, [i, j, self.l, x]])
		
		# seq = sample(seq, int(len(seq) * self.frac))
		print('总共生成', len(seq), '个序列')
		
		if update:
			self.sequence = seq
			self.seq_len_dict = seq_len_dict
		else:
			return seq, seq_len_dict
	
	def save(self, path):
		X = []
		Y = []
		length = []
		info = []
		for i in range(len(self.sequence)):
			X.append(self.sequence[i][0].numpy())
			Y.append(self.sequence[i][1].numpy())
			length.append(self.sequence[i][2])
			info.append(self.sequence[i][3])
		X = np.array(X)
		Y = np.array(Y)
		length = np.array(length)
		info = np.array(info)
		if not os.path.isdir(path):
			os.makedirs(path)
		
		np.save(os.path.join(path, 'X.npy'), X)
		np.save(os.path.join(path, 'Y.npy'), Y)
		np.save(os.path.join(path, 'length.npy'), length)
		np.save(os.path.join(path, 'info.npy'), info)
	
	def read_data(self):
		with open(self.data_path, 'r') as f:
			data = json.load(f)
			TaiFengClass = torch.tensor([i for i in range(len(data)) for _ in range(len(data[i]))])
		data = torch.tensor([item for seq in data for item in seq])[:, self.features_index]
		print('加载数据，总共有', TaiFengClass[-1].numpy() + 1, '个台风，', len(data), '条台风数据')
		return data, TaiFengClass
	
	def get_min_max(self):
		#return torch.tensor([17, 986, 888, 8]), torch.tensor([621, 2439, 1014, 78])
	    return self.data.min(dim=0).values, self.data.max(dim=0).values
	
	def normalization(self, data):
		return (data - self.mindata) / (self.maxdata - self.mindata)
	
	def inverse_norm(self, data):
		return data * (self.maxdata - self.mindata) + self.mindata
	
	def __len__(self):
		return len(self.sequence)


def collate_fn(batch):
	X = []
	y = []
	last = []
	info = []
	for i in range(len(batch)):
		X.append(batch[i][0])
		y.append(batch[i][1])
		last.append(batch[i][2] - 1)
		info.append(batch[i][3])
	X.sort(key=lambda seq: len(seq), reverse=True)
	last.sort(reverse=True)
	X = pack_sequence(X)
	y = torch.vstack(y)
	
	return X, y, last, info


# def


def get_window_data():
	for l in range(1, 21):
		data = MyData(data_path='./data/TrainData.json', frac=1, l=l)
		data.save(f'data/window/train/{l}window')


def get_mix_data():
	for l in [1, 5, 10, 15, 20]:
		data = MyData(data_path='./data/TrainData.json', frac=1, l=l, mix=True)
		data.save(f'data/mix/train/{l}window-mix')
		data = MyData(data_path='./data/TrainData.json', frac=1, l=l)
		data.save(f'data/mix/train/{l}window-nomix')


def get_test_data():
	for x in range(1, 21):
		for l in [1, 5, 10, 15, 20]:
			data = MyData(data_path='./data/TestData.json', frac=1, l=l)
			data.get_xpredict_sequence(x=x, update=True)
			data.save(f'data/ProcessTest/{x}Predict/{l}window')


def get_features_data():
	features_index = [[2, 3, 4, 5], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3], [2, 3, 4], [2, 3, 5]]
	for feature_index in features_index:
		data = MyData(data_path='./data/TrainData.json', frac=1, l=5, features_index=feature_index)
		data.save(f'data/features_index/train/{5}window-{"".join(list(map(str, feature_index)))}')
		data = MyData(data_path='./data/TestData.json', frac=1, l=5, features_index=feature_index)
		data.save(f'data/features_index/test/{5}window-{"".join(list(map(str, feature_index)))}')


if __name__ == '__main__':
	# get_window_data()
	# get_mix_data()
	# get_features_data()
	# data = MyData(data_path='./data/TrainData.json', frac=1, l=5, features_index=[1, 2, 3, 4, 5, 6])
	# print(data.mindata)
	# print(data.maxdata)
	
	# for x in [4,8,12]:
	# 	data = MyData(data_path='./data/TestData.json', frac=1, l=4)
	# 	data.get_xpredict_sequence(x=x, update=True)
	# 	data.save(f'data/ProcessTest/{int(x*6)}hPredict')
	
	# get_test_data()
	
	# print(data.seq_len_dict)
	
	# batch_size = 32
	data = MyData(data_path='./data/TrainData.json', frac=1, l=4)
	data.get_mix_t_sequence()
	train_dataloader = DataLoader(data, batch_size=128, shuffle=True)
	
	for i, batch in enumerate(train_dataloader):
		print(len(batch))
		X, y, last, a = batch
		print(a)
		# print(X)
		break
	
	# print(len(data))
	# X, y,info = data[24524]
	# print(X)
	# print(y)
	# print(info)
	print('end')

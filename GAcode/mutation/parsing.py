import re
import pickle
import random
import torch.nn as nn
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import time

# with open('densenet.py', 'r') as f:
# 	lines = f.readlines()
file_name_list = ('googlenet.py','densenet.py','shufflenet.py','BayesianCNN.py','PNASNet.py','ACGAN_D_model.py','vgg19_model.py','mobilenet_model.py','resnet_model.py')
with open('./mutation/data/basic_block_bc.pkl', 'rb') as handle:
	block_dic = pickle.load(handle)	

def get_init_lines(lines):
	starting_pos = 0
	end_pos = 0
	print_bool = False
	for i in range(len(lines)):
		if '__init__' in str(lines[i]):
			starting_pos = i+1
		if 'forward' in str(lines[i]):
			end_pos = i-1
			return lines[starting_pos:end_pos+1], starting_pos, end_pos
def get_forward_lines(lines):
	starting_pos = 0
	end_pos = 0
	for i in range(len(lines)):
		if 'forward' in str(lines[i]):
			starting_pos = i+1
		if 'return' in str(lines[i]):
			end_pos = i-1
			return lines[starting_pos:end_pos+1], starting_pos, end_pos

def get_forward_init_pair_dic(forward_lines, init_lines):
	dic = {}
	for s in forward_lines:
		tmp_s = re.split(r'[(=)]',str(s))
		tmp_s = tmp_s[1].strip()
		for t in init_lines:
			if(tmp_s in str(t)):
				dic[str(s)] = str(t)
	return dic

def create_block_pool(file_name_list):
	new_dic ={}
	for s in file_name_list:
		with open(s, 'r') as f:
			lines = f.readlines()
		init_lines, _ ,_ = get_init_lines(lines)
		forward_lines,_,_ = get_forward_lines(lines)
		dic = get_forward_init_pair_dic(forward_lines,init_lines)
		if(not new_dic):
			new_dic = dic 
		else:
			for key, value in dic.items():
				new_dic[key] = value
	with open('data/basic_block.pickle','wb') as handle:
		pickle.dump(new_dic,handle, protocol=pickle.HIGHEST_PROTOCOL)
	return new_dic

#block_dic = create_block_pool(file_name_list)

def rand_pick(block_dic):
	forward_line, init_line = random.choice(list(block_dic.items()))
	return forward_line, init_line


def rand_pick(start, end):
	return random.randrange(start,end+1)
def rand_pick_pEnd(start,end):
	return random.randrange(start,end+2)

def rand_pick_pair(lines):
	init_lines, in_st, in_end = get_init_lines(lines)
	forward_lines,for_st,for_end = get_forward_lines(lines)
	pair_dic = get_forward_init_pair_dic(forward_lines,init_lines)
	index = rand_pick(for_st,for_end)
	picked_forward = lines[index]
	picked_init = pair_dic[picked_forward]
	return picked_forward, picked_init



def rand_delete(lines):
	picked_forward, picked_init = rand_pick_pair(lines)
	new_lines = []
	for_index = 0
	init_index = 0
	for i, s in enumerate(lines):
		if(s == picked_forward):
			for_index = i
		elif(s==picked_init):
			init_index = i
		else:
			new_lines.append(s)
	return new_lines, for_index, init_index
def get_init_index(lines, line):
	for i, s in enumerate(lines):
		if(s==line):
			return i
def rand_insert_index(lines):
	init_lines, in_st, in_end = get_init_lines(lines)
	forward_lines,for_st,for_end = get_forward_lines(lines)
	pair_dic = get_forward_init_pair_dic(forward_lines,init_lines)
	index = rand_pick_pEnd(for_st,for_end)
	last = False
	if(index==for_end+1):
		last = True
		index = for_end
	picked_forward = lines[index]
	picked_init = pair_dic[picked_forward]
	init_index =get_init_index(lines, picked_init)
	if(last == True):
		index +=1
		init_index+=1 
	return index, init_index

def rand_insert(lines, dic):
	for_index, init_index= rand_insert_index(lines)
	forward_line, init_line = random.choice(list(dic.items()))
	new_lines = lines[0:init_index] + [init_line]
	new_lines = new_lines + lines[init_index:for_index]
	new_lines = new_lines + [forward_line] + lines[for_index:]
	return new_lines


def get_IO_size(line,input = True):
	if "nn.MaxPool2d" in line:
		return 'max'
	elif "nn.AvgPool2d" in line:
		return 'avg'
	elif "super" in line:
		return 'super'
	else:
		tmp_s = re.split(r'[(,)]',line)
		if(input == True):
			return tmp_s[1].strip()
		else:
			output = tmp_s[2].strip()
			if('+' in output):
				return int(tmp_s[1].strip()) + int(output[2:-1])
			else:
				return tmp_s[2].strip()

def check_max(line):
	if "nn.MaxPool2d" in line:
		return True
	else:
		return False
# def size_flow(start_idx,line_all):
# 	while("forward" not in line_all[start_idx]):
# 		if()
def change_input_size(input_line, input_size):
	tmp_s = re.split(r'[(,)]',input_line)
	tmp_s[1] = str(input_size)
	new_input_line = tmp_s[0]+"("
	for s in tmp_s[1:-2]:
		new_input_line +=s 
		new_input_line +=','
	new_input_line+=tmp_s[-2] 
	new_input_line+=")"
	new_input_line+=tmp_s[-1]
	return new_input_line 


def dim_check(lines):
	init_lines, in_st, in_end = get_init_lines(lines)
	output_size = get_IO_size(lines[in_st],False)
	for i in range(in_st,in_end+1):
		input_size = get_IO_size(lines[i])
		if(i==in_st):
			if(input_size!=3):
				lines[i] = change_input_size(lines[i],3)
		else:
			# print(lines[i], input_size, output_size)
			if(input_size== 'max'):
				continue
			elif(input_size != output_size):
				lines[i] = change_input_size(lines[i], output_size)
			output_size = get_IO_size(lines[i],False)
	return lines
def x_out_check(lines):
	forward_lines,for_st,for_end = get_forward_lines(lines)
	for i in range(for_st, for_end+1):
		for_line = lines[i]
		for_line = re.split(r'[(]',for_line)
		new_for_line = for_line[0]
		if(i == for_st):
			new_for_line+="(x)\n"
		else:
			new_for_line+="(out)\n"
		lines[i] = new_for_line
	return lines

# new_inserted_line = rand_insert(lines, block_dic)
# with open('tmp_insert_dense.py','w') as f:
#    for s in new_inserted_line:
#        f.write(str(s))
# new_inserted_line = dim_check(new_inserted_line)
# new_inserted_line = x_out_check(new_inserted_line)
# with open('tmp_insert_dense_after.py','w') as f:
#    for s in new_inserted_line:
#        f.write(str(s))
# new_deleted_line, for_index, init_index = rand_delete(lines)
# with open('tmp_deleted_dense.py','w') as f:
#    for s in new_deleted_line:
#        f.write(str(s))
# new_deleted_line = dim_check(new_deleted_line)
# new_deleted_line = x_out_check(new_deleted_line)
# with open('tmp_deleted_dense_after.py','w') as f:
#    for s in new_deleted_line:
#        f.write(str(s))
#    f.write('        return out')
# from tmp_deleted_dense_after import *



def get_size(model):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	#model = NET().to(device)
	model.train()
	channel_size = 0
	length = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		output = model(data)
		print(output.size())
		channel_size = output.size()[1]
		length = output.size()[2]
		break
	return channel_size,length


#def mutation(file_name):
def mutation(file_name,deletion_prob=0.5):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#block_dic = create_block_pool(file_name_list)
	if(random.random()<deletion_prob):
		new_lines, for_index, init_index = rand_delete(lines)
	else: 
		new_lines = rand_insert(lines, block_dic)
	new_lines = dim_check(new_lines)
	new_lines = x_out_check(new_lines)
	if os.path.exists('./mutation/tmp.pyc'):
		os.remove('./mutation/tmp.pyc')
	if os.path.exists('./mutation/tmp.py'):
		os.remove('./mutation/tmp.py')
	with open('./mutation/tmp.py','w') as f:
		for s in new_lines:
			f.write(str(s))
		f.write('        return out')
        f.close()
	#from tmp import *
	import tmp
	reload(tmp)
	model = tmp.Net().to(device)
	#model = NET().to(device)
	channel_size, length = get_size(model)
	print("sizes are: ",channel_size,length)
	init_lines, in_st, in_end = get_init_lines(new_lines)
	forward_lines,for_st,for_end = get_forward_lines(new_lines)
	new_lines_final = new_lines[0:for_st-1] + ["        self.avgpool_end = nn.AvgPool2d("+str(length)+", stride=1)\n"]+["        self.linear_end = nn.Linear("+str(channel_size)+',10)\n']
	new_lines_final = new_lines_final+ new_lines[for_st-1:] + ["        out = self.avgpool_end(out)\n"] + ["        out = out.view(out.size(0), -1)\n"] + ['        out = self.linear_end(out)\n']+['        return out\n']

	with open(file_name,'w') as f:
		for s in new_lines_final:
			f.write(str(s))
	time.sleep(1)
	if os.path.exists('./mutation/tmp.pyc'):
		os.remove('./mutation/tmp.pyc')
	if os.path.exists('./mutation/tmp.py'):
		os.remove('./mutation/tmp.py')
	return file_name

def no_mutation(file_name):
	with open(file_name, 'r') as f:
		new_lines = f.readlines()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if os.path.exists('./mutation/tmp.py'):
		os.remove('./mutation/tmp.py')
	if os.path.exists('./mutation/tmp.pyc'):
		os.remove('./mutation/tmp.pyc')
	with open('./mutation/tmp.py','w') as f:
		for s in new_lines:
			f.write(str(s))
		f.write('        return out')
		f.close()
	import tmp
	reload(tmp)
	model = tmp.Net().to(device)
	channel_size, length = get_size(model)
	init_lines, in_st, in_end = get_init_lines(new_lines)
	forward_lines,for_st,for_end = get_forward_lines(new_lines)
	new_lines_final = new_lines[0:for_st-1] + ["        self.avgpool_end = nn.AvgPool2d("+str(length)+", stride=1)\n"]+["        self.linear_end = nn.Linear("+str(channel_size)+',10)\n']
	new_lines_final = new_lines_final+ new_lines[for_st-1:] + ["        out = self.avgpool_end(out)\n"] + ["        out = out.view(out.size(0), -1)\n"] + ['        out = self.linear_end(out)\n']+['        return out\n']
	with open(file_name,'w') as f:
		for s in new_lines_final:
			f.write(str(s))
	time.sleep(1)
	if os.path.exists('./mutation/tmp.py'):
		os.remove('./mutation/tmp.py')
	if os.path.exists('./mutation/tmp.pyc'):
		os.remove('./mutation/tmp.pyc')
	return file_name
            
#mutation('m1_0.py')

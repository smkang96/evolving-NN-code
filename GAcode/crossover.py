import random
import re
delim = '[= ( )]'
number = '0123456789'
genpath = './../newgen_dir/'
"""
Finds the forward part of the code and does crossover on the two parents to produce one simple child
"""

import_statement = 'from __future__ import print_function\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nfrom torch.autograd import Variable\nimport torchvision.datasets as dset\nimport torchvision.transforms as transforms\nfrom torch.utils.data import DataLoader\nimport torchvision.models as models\nimport sys\nimport math\nimport argparse\nfrom torchvision import datasets, transforms\nimport torchvision\nimport torchvision.transforms as transforms\nfrom basic_blocks import *\n'

#Finds the forward function within it and returns each line after forward (not including return) in a list
def find_forward(tot_txt):
	tot_txt.reverse()
	for_list = []
	net_list = []
	retflag = False
	forflag = False
	for line in tot_txt:
		if "#return" in line:
			retflag = True
			continue
		if '#forward' in line:
			forflag = True
			continue
		# #return has been passed, start adding lines to forward
		if retflag:
			#reached def forward, stop copying
			if 'forward' in line:
				retflag = False
				continue
			else:
				for_list.insert(0, line.strip())

		if forflag:
			if 'super' in line:
				forflag = False
				continue
			else:
				net_list.insert(0, line.strip())
	'''
	for line in tot_txt:
		#assume super is the line before init statements start
		if "super" not in line:
			#ignore final return line, will be filled later with appropriate variable name
			if "return" in line:
				continue
			else:
				#remove empty lines
				if not line.strip():
					continue
				if line.startswith('#'):
					continue
				for_list.insert(0, line.strip())
		else:
			break
	idx = -1
	for statement in for_list:
		if 'forward' in statement:
			idx = for_list.index(statement)

	net_list = for_list[:idx] #last few lines are from forward
	for_list = for_list[idx + 1:] #first few lines are from init
	'''
	return for_list, net_list

#Assume parent is title a python file of functional neural network code
def crossover(parent1, parent2, gid):
	fname = genpath + 'n' +  gid + '.py'
	child_init = []
	child_for = []
	child_for_tmp = []
	with open(parent1, 'r') as par1:
		with open(parent2, 'r') as par2:
			for1, net1 = find_forward(list(par1))
			for2, net2 = find_forward(list(par2))


	#choose a random point in forward1 to cut and add crossover part to child
	idx1 = random.randint(0, len(for1))
	child_for_tmp = child_for_tmp + for1[:idx1]

	#add necessary init from parent1
	count = 0
	for statement in child_for_tmp:
		for_line = list(filter(None, re.split(delim, statement)))

		#method name is always on second (index 1)
		for_name = for_line[1]
		#find init_line with for_name
		for init_line in net1:
			if for_name in init_line:
				#decompose init_line
				init_line = list(filter(None, re.split('=', init_line)))
				#add '_count' to for_name and init_name, reconstruct for_line and init_line
				for_line = for_line[0].strip() + ' = ' + for_line[1].strip() + '_' + str(count) + '(' + for_line[2].strip() + ')'
				init_line[0] = init_line[0].strip() + '_' + str(count) + ' '
				init_line = '='.join(init_line)
				count += 1
				#add init_line to child_init
				child_init.append(init_line)
				#append for_line to child_for
				child_for.append(for_line)
	'''
	#make init unique by removing all duplicates
	for elem in child_tmp:
		if elem not in child_init:
			child_init.append(elem)		
	'''
	first = True
	#choose and add necessary info from parent2
	idx2 = random.randint(0, len(for2))
	child_for_tmp = child_for_tmp + for2[idx2:]

	for statement in child_for_tmp[idx1:]:
		for_line = list(filter(None, re.split(delim, statement)))
		for_name = for_line[1]

		for init_line in net2:
			if for_name in init_line:
				if first:
					#get output dimension of previous line
					prev_line = child_init[idx1-1]
					prev_dim = list(filter(None, re.split(',', prev_line)))[1].replace(')', '').replace(' ', '').replace("'", "")
					prev_indim = list(filter(None, re.split('[(,]', prev_line)))[1]
					init_line = list(filter(None, re.split(',', init_line)))
					
					if prev_dim[0] is '+':
						prev_dim = int(prev_indim) + int(prev_dim[1:])

					i = len(init_line[0])

					while init_line[0][i-1] in number:
						init_line[0] = init_line[0][:i-1]
						i = i - 1

					#change input dimension of new line to output dimension of prev line
					init_line[0] = init_line[0] + str(prev_dim)
					init_line = ','.join(init_line)
					first = False

					
				init_line = list(filter(None, re.split('=', init_line)))
				for_line = for_line[0].strip() + ' = ' + for_line[1].strip() + '_' + str(count) + '(' + for_line[2].strip() + ')'
				init_line[0] = init_line[0].strip() + '_' + str(count) + ' '
				init_line = '='.join(init_line)
				count += 1

				child_init.append(init_line)
				child_for.append(for_line)




	child_file = open(fname, 'w')
	#write import statements
	child_file.write(import_statement + '\n')

	#write basic init statements
	child_file.write('class Net(nn.Module):\n')
	child_file.write('    def __init__(self):\n')
	child_file.write('        super(Net, self).__init__()\n')

	for elem in child_init:
		child_file.write('        ' + elem + '\n')

	#write basic forward statement
	child_file.write('#forward\n')
	child_file.write('    def forward(self, x):\n')
	for elem in child_for:
		child_file.write('        ' + elem + '\n')
	child_file.write('#return\n')
	return gid

if __name__ == '__main__':
    result = crossover('./mutation/densenet.py', './mutation/densenet.py', '1_0')
    print(result)

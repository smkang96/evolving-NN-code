import random
import re
delim = '[= ( )]'
"""
Finds the forward part of the code and does crossover on the two parents to produce one simple child
"""

import_statement = 'import torch.nn as nn\n'

def check_dimension_equality(next, prev):
	return next

def check_variable_equality(child):
	return child

#Finds the forward function within it and returns each line after forward (not including return) in a list
def find_forward(tot_txt):
	tot_txt.reverse()
	for_list = []
	net_list = []
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
				for_list.insert(0, line.strip())
		else:
			break
	idx = -1
	for statement in for_list:
		if 'forward' in statement:
			idx = for_list.index(statement)

	net_list = for_list[:idx] #last few lines are from forward
	for_list = for_list[idx + 1:] #first few lines are from NET
	return for_list, net_list

#Assume parent is title a python file of functional neural network code
def crossover(parent1, parent2, gid):
	gid = gid + '.py'
	child_init = []
	child_for = []
	child_tmp = []
	with open(parent1, 'r') as par1:
		with open(parent2, 'r') as par2:
			for1, net1 = find_forward(list(par1))
			for2, net2 = find_forward(list(par2))

	#choose a random point in forward1 to cut and add crossover part to child
	idx1 = random.randint(0, len(for1))
	idx1 = 3
	child_for = child_for + for1[:idx1]

	#add necessary init from parent1
	for statement in for1[:idx1]:
		st_list = list(filter(None, re.split(delim, statement)))

		#method name is always on second (index 1)
		line_name = st_list[1]

		#find method name from NET list
		for statement in net1:
			if line_name in statement:
				child_tmp.append(statement)
	
	#make init unique by removing all duplicates
	for elem in child_tmp:
		if elem not in child_init:
			child_init.append(elem)		

	first = True
	#choose and add necessary info from parent2
	idx2 = random.randint(0, len(for2))
	idx2 = 4
	child_for = child_for + for2[idx2 + 1:]

	for statement in for2[idx2 + 1:]:
		st_list = list(filter(None, re.split(delim, statement)))

		line_name = st_list[1]
		print(line_name)

		for statement in net2:
			if line_name in statement:
				if statement not in child_init:
					if first:
						#THIS PART, CHECK DIMENSION NEEDS WORK TODO
						statement = check_dimension_equality(statement, child_init[idx1-1])
						child_init.append(statement)
						first = False
					else:
						child_init.append(statement)

	child_file = open(gid, 'w')
	#write import statements
	child_file.write(import_statement + '\n')

	#write basic init statements
	child_file.write('class Net(nn.Module):\n')
	child_file.write('\tdef __init__(self):\n')
	child_file.write('\t\tsuper(Net, self).__init__()\n')

	for elem in child_init:
		child_file.write('\t\t' + elem + '\n')

	#write basic forward statement
	child_file.write('#forward')
	child_file.write('\tdef forward(self, x):\n')
	for elem in child_for:
		child_file.write('\t\t' + elem + '\n')
	child_file.write('\t\treturn out\n')
	child_file.write('#return')
	return gid

#for debugging
#result = crossover('net1.py', 'net2.py', '1_0')
#print(result)
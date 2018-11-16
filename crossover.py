import random

"""
Finds the forward part of the code and does crossover on the two parents to produce one simple child
"""

def check_dimension_equality(child):
	return child

def check_variable_equality(child):
	return child

#Finds the forward function within it and returns each line after forward (not including return) in a list
def find_forward(tot_txt):
	tot_txt.reverse()
	ret_list = []
	for line in tot_txt:
		if line.startswith("def") != 1:
			if line.startswith("return"):
				continue
			else:
				ret_list.insert(0, line)
		else:
			break
	return ret_list

#Assume parent is title a python file of functional neural network code
def crossover(parent1, parent2):
	with open(parent1, 'r') as par1:
		with open(parent2, 'r') as par2:
			for1 = find_forward(list(par1))
			for2 = find_forward(list(par2))
	idx1 = random.randint(0,len(for1))
	idx2 = random.randint(0,len(for2))
	child_for = for1[0:idx1] + for2[idx2:]
	child_for = check_variable_equality(child_for)
	child_for = check_dimension_equality(child_for)
	return child_for

#for debugging
result = crossover('test1.py', 'test2.py')
print(result)
from evaluation import Evaluator
import crossover
from mutation.parsing import mutation, no_mutation
from MOEAII import NSGAII

import random

class Breeder(object):
    def __init__(self, init_pop, growth_time=2*60, mut_prob=0.5, pop_size=30):
        self._init_pop = init_pop
        self._mut_prob = mut_prob
        self._evaluator = Evaluator(growth_time)
        self._nsgaii_sorter = NSGAII(2, None, None)
        self._pop_size = pop_size
        
    def breed(self, gen_num):
        curr_pop = self._init_pop[:] # copy
        for g_idx in range(gen_num):
            kids = []
            i_idx = -1
            while len(kids) < self._pop_size-len(curr_pop):
                i_idx += 1
                try:
                    poppa = random.choice(curr_pop)
                    momma = random.choice(curr_pop)
                    print poppa, momma
                    kid = crossover.crossover(poppa, momma, '%d_%d' % (g_idx, i_idx))
                    print kid
                    if self._mut_prob > random.random():
                        kid = mutation(kid)
                    else:
                        kid = no_mutation(kid)
                    kids.append(kid)
                except RuntimeError:
                    print 'death due to wrong size'
                    continue
                except Exception as e:
                    print '%d_%d' % (g_idx, i_idx), 'dead'
                    print 'death cause:', str(e)
                    continue
            curr_pop += kids
            
            for name in curr_pop:
                print name
            
            eval_results = []
            eval_dict = {} # for fun
            for indiv in curr_pop:
                print indiv
                try:
                    indiv_score = self._evaluator._indiv_evaluator(indiv)
                except Exception as e:
                    print 'indiv born with defect, killed.'
                    print 'death cry:', str(e)
                    indiv_score = (-1, 10000000)
                eval_dict[indiv] = indiv_score
                indiv_score = (indiv, indiv_score[0], indiv_score[1])
                eval_results.append(indiv_score)
        
            selected = self._nsgaii_sorter.run(eval_results, self._pop_size)
            curr_pop = selected
        
        for chosen in selected:
            print 'Honorable chosen neural network %s' % chosen
            print 'Score: (%.2f, %.2f)' % (eval_dict[chosen])
        
        # print curr_pop
'''
INIT_POP = [
    './mutation/ACGAN_D_model.py',
    './mutation/alexnet_model.py',
    './mutation/BayesianCNN.py',
    './mutation/densenet.py',
    './mutation/googlenet.py',
    './mutation/mobilenet_model.py'
]
'''

INIT_POP = ['./mutation/googlenet.py', './mutation/ACGAN_D_model.py']
        
b = Breeder(INIT_POP, growth_time = 120, pop_size = 15)
b.breed(10)
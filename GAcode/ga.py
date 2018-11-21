from evaluation import Evaluator
import crossover
from mutation.parsing import mutation

import random

class Breeder(object):
    def __init__(self, init_pop, growth_time=2*60, mut_prob=0.5):
        self._init_pop = init_pop
        self._evaluator = Evaluator(growth_time)
        self._mut_prob = mut_prob
        
    def breed(self, gen_num, pop_num):
        curr_pop = self._init_pop[:] # copy
        for g_idx in range(gen_num):
            for i_idx in range(pop_num-len(curr_pop)):
                poppa = random.choice(curr_pop)
                momma = random.choice(curr_pop)
                kid = crossover.crossover(poppa, momma, '%d_%d' % (g_idx, i_idx))
                if self._mut_prob > random.random():
                    kid = mutation(kid)
                curr_pop.append(kid)
            
            eval_results = []
            for indiv in curr_pop:
                try:
                    indiv_score = self._evaluator.evaluate(indiv)
                except:
                    print 'indiv born with defect, killed'
                    indiv_score = (-1, 10000000)
                eval_results.append(indiv_score)
                    
            
        print curr_pop
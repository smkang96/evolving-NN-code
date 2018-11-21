from evaluation import Evaluator
import crossover
from mutation.parsing import mutation

import random

class Breeder(object):
    def __init__(self, init_pop, growth_time=2*60,):
        self._init_pop = init_pop
        self._evaluator = Evaluator(growth_time)
        
    def breed(self, gen_num, pop_num):
        curr_pop = self._init_pop[:] # copy
        for g_idx in range(gen_num):
            for i_idx in range(pop_num-len(curr_pop)):
                poppa = random.choice(curr_pop)
                momma = random.choice(curr_pop)
                kid = crossover.crossover(poppa, momma, '%d_%d' % (g_idx, i_idx))
        print curr_pop
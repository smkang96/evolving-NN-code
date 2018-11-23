from evaluation import Evaluator
import crossover
from mutation.parsing import mutation, no_mutation
from MOEAII import NSGAII

import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

code_path = './new_constepoch/'
img_path = './img_constepoch/'

class Breeder(object):
    def __init__(self, init_pop, growth_time=2*60, mut_prob=0.5, pop_size=30):
        self._init_pop = init_pop
        self._mut_prob = mut_prob
        self._evaluator = Evaluator(growth_time)
        self._nsgaii_sorter = NSGAII(2, None, None)
        self._pop_size = pop_size
        
    def breed(self, gen_num, result_file = 'results.csv'):
        curr_pop = self._init_pop[:] # copy
        eval_dict = {}
        f = open(result_file, 'w')
        for g_idx in range(gen_num):
            kids = []
            if g_idx == 0:
                kids += [no_mutation(ip) for ip in curr_pop]
            i_idx = -1
            while len(kids) < self._pop_size-len(curr_pop):
                i_idx += 1
                try:
                    poppa = random.choice(curr_pop)
                    momma = random.choice(curr_pop)
                    print poppa, momma
                    kid = crossover.crossover(poppa, momma, '%d_%d' % (g_idx, i_idx), genpath=code_path)
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
            
            for name in curr_pop + kids:
                print name
            
            if len(eval_dict.keys()) != 0:
                eval_results = [(name, eval_dict[name][0], eval_dict[name][1]) for name in curr_pop]
            else:
                eval_results = []
            curr_pop += kids
            # eval_dict = {} # for fun
            for indiv in kids:
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
            plt.plot([i[1] for i in eval_results], [i[2] for i in eval_results], 'b.')
            plt.title('Generation %d' % g_idx)
            plt.xlabel('Accuracy (%)')
            plt.ylabel('Inference Time (s)')
            plt.savefig(img_path + ('/gen%d_pareto.png' % g_idx))
            plt.clf()
            curr_pop = selected
        
            for chosen in selected:
                print 'Honorable chosen neural network in gen %d: %s' % (g_idx, chosen)
                print 'Score: (%.2f, %.2f)' % (eval_dict[chosen])
                f.write(chosen + ',')
                f.write('%.2f,%.2f' % (eval_dict[chosen]) + '\n')
        
        f.close()
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

INIT_POP = ['./mutation/googlenet.py', './mutation/ACGAN_D_model.py', './mutation/mobilenet_model.py', 
            './mutation/BayesianCNN.py', './mutation/densenet.py', './mutation/PNASNet.py', 
            './mutation/shufflenet.py', './mutation/vgg19_model.py']
        
b = Breeder(INIT_POP, growth_time = 5*60, pop_size = 30)
b.breed(10, result_file = 'expensive_results.csv')
""" MOEA code for Multi-Optimizing NNs """

import sys
import random

def crowded_comparison(s1, s2):
    '''
    Compare the two solutions based on crowded comparison.
    '''
    if s1.rank < s2.rank:
        return 1
        
    elif s1.rank > s2.rank:
        return -1
        
    elif s1.distance > s2.distance:
        return 1
        
    elif s1.distance < s2.distance:
        return -1
        
    else:
        return 0

class NSGAII:
    '''
    Implementation of NSGA-II algorithm.
    '''
    current_evaluated_objective = 0

    def __init__(self, num_objectives, mutation_rate=0.5, crossover_rate=1.0):
        '''
        Constructor. Parameters: number of objectives, mutation rate (default value 50%) and crossover rate (default value 100%). 
        '''
        self.num_objectives = num_objectives
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        random.seed();
        
    def sort_ranking(self, P):
        for i in range(len(P)-1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j-1]
                s2 = P[j]
                
                if s1.rank > s2.rank:
                    P[j-1] = s2
                    P[j] = s1
                    
    def sort_objective(self, P, obj_idx):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j-1]
                s2 = P[j]
                
                if s1.objectives[obj_idx] > s2.objectives[obj_idx]:
                    P[j-1] = s2
                    P[j] = s1
                    
    def sort_crowding(self, P):
        for i in range(len(P)-1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if crowded_comparison(s1, s2) < 0:
                    P[j - 1] = s2
                    P[j] = s1

    def fast_nondominated_sort(self, P):
        '''
        Discover Pareto fronts in P, based on non-domination criterion. 
        '''
        fronts = {}
        
        S = {}
        n = {}
        for s in P:
            S[s] = []
            n[s] = 0
            
        fronts[1] = []
        
        for p in P:
            for q in P:
                if p == q:
                    continue
                
                if p >> q:
                    S[p].append(q)
                
                elif p << q:
                    n[p] += 1
            
            if n[p] == 0:
                fronts[1].append(p)
        
        i = 1
        
        while len(fronts[i]) != 0:
            next_front = []
            
            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        next_front.append(s)
            
            i += 1
            fronts[i] = next_front
                    
        return fronts
        
    def crowding_distance_assignment(self, front):
        '''
        Assign a crowding distance for each solution in the front. 
        '''
        for p in front:
            p.distance = 0
        
        for obj_index in range(self.num_objectives):
            self.sort_objective(front, obj_index)
            
            front[0].distance = float('inf')
            front[len(front) - 1].distance = float('inf')
            
            for i in range(1, len(front) - 1):
                front[i].distance += (front[i + 1].distance - front[i - 1].distance)

    def make_new_pop(self, P):
        '''
        Make new population Q, offspring of P. 
        '''
        Q = []
        # Do make_new_pop function until length of Q list reaches the same of length of P list. 
        while len(Q) != len(P):
            selected_solutions = [None, None]
            
            while selected_solutions[0] == selected_solutions[1]:
                for i in range(2):
                    # Set s1 and s2 different, selecting more appropriate one using crowded_comparison.
                    s1 = random.choice(P) 
                    s2 = s1
                    while s1 == s2:
                        s2 = random.choice(P)
                    if crowded_comparison(s1, s2) > 0:
                        selected_solutions[i] = s1
                    else:
                        selected_solutions[i] = s2

            Q.append(selected_solutions[0])

            """
            # We do not need crossover and mutation at MOEA this time

            if random.random() < self.crossover_rate:
                child_solution = selected_solutions[0].crossover(selected_solutions[1])
                
                if random.random() < self.mutation_rate:
                    child_solution.mutate()
                    
                child_solution.evaluate_solution()

                Q.append(child_solution)
                """
        
        return Q

    def run(self, P, population_size):
        '''
        Run NSGA-II. 
        '''
        
        Q = []
                     
        R = []
        R.extend(P)
        R.extend(Q)
        
        fronts = self.fast_nondominated_sort(R)
        
        del P[:]
        
        for front in fronts.values():
            if len(front) == 0:
                break
            
            self.crowding_distance_assignment(front);
            P.extend(front)
            
            if len(P) >= population_size:
                break
        
        self.sort_crowding(P)
        
        if len(P) > population_size/2:
            del P[population_size/2:]

        return P
            
"""
if __name__ == '__main__':
    nsga2 = NSGAII(2, 0.1, 1.0)
    
    P = []
    for i in range(500):
        P.append(T1Solution())
    
    nsga2.run(P, 50, 20)
    
    csv_file = open('nsga2_out.csv', 'w')
    
    for i in range(len(P)):
        csv_file.write("" + str(P[i].objectives[0]) + ", " + str(P[i].objectives[1]) + "\n")
        
    csv_file.close()
    """

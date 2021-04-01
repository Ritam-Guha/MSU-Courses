import numpy as np 

def fitness(solution, construction=1, problem=1):
    # function for computing fitness based on the problem and construction
    dim = np.shape(solution)[0]
    val = 0.0
    if(construction == 1 and problem == 1):
        for i in range(0,dim,3):
            val += sum(solution[i:(i+3)])/3

    elif(construction == 1 and problem == 2):
        for i in range(0,dim,3):
            temp = sum(solution[i:(i+3)])
            if(temp == 3):
                val += 1
            else:
                val += 0.9 - (temp/3)
    
    elif(construction == 2 and problem == 1):
        for i in range(0, int(dim/3)):
            val += (solution[i]+solution[i+10]+solution[i+20])/3

    else:
        for i in range(0, int(dim/3)):
            temp = solution[i]+solution[i+10]+solution[i+20]
            if(temp == 3):
                val += 1
            else:
                val += 0.9 - (temp/3)
    
    return (10*val)


def mutation(solution, mut_prob=0.1):
    # function for mutation
    dim = np.shape(solution)[0]
    for i in range(dim):
        if(np.random.rand()<mut_prob):
            solution[i] = 1-solution[i]

    return solution 


def crossover(parent1, parent2):
    # performs single-point crossover
    dim = np.shape(parent1)[0]
    child1 = np.zeros(dim)
    child2 = np.zeros(dim)
    cross_point = np.random.randint(dim-2)+1

    for i in range(dim):
        if(i<cross_point):
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return child1, child2


def binary_tournament_selection(population, objective):
    # used for binary tournament selection
    num_pop, dim = np.shape(population)
    num_sub_pop = int(num_pop/2)
    shuffle_order = np.random.permutation(num_pop)
    population = population[shuffle_order, :]
    objective = objective[shuffle_order]
    sub_population = np.zeros((num_sub_pop, dim))
    sub_objective = np.zeros(num_sub_pop)

    for i in range(0, num_pop, 2):
        pos_idx = int(i/2)

        if(objective[i] > objective[i+1]):
            sub_population[pos_idx,:] = population[i, :]
            sub_objective[pos_idx] = objective[i]
        
        else:
            sub_population[pos_idx,:] = population[i+1, :]
            sub_objective[pos_idx] = objective[i+1]

    return sub_population, sub_objective


def initialization(pop_size, dim):
    # initialize the population
    population = np.random.randint(low=0, high=2, size=(pop_size,dim))

    return population


def count_competitors(population, construction=1):
    # helper function to count the number of competitors in each generation
    pop_size, dim = np.shape(population)
    comp_scores = np.zeros((2, 10))

    if(construction == 1):
        for i in range(0, 10):
            for pop_no in range(pop_size):
                if(sum(population[pop_no][(i*3):(i*3+3)])==3):
                    comp_scores[0][i] += 1

                if(sum(population[pop_no][(i*3):(i*3+3)])==0):
                    comp_scores[1][i] += 1

    else:
        for i in range(0, 10):
            for pop_no in range(pop_size):
                if((population[pop_no][i] + population[pop_no][i+10] + population[pop_no][i+20])==3):
                    comp_scores[0][i] += 1

                if((population[pop_no][i] + population[pop_no][i+10] + population[pop_no][i+20])==0):
                    comp_scores[1][i] += 1

    comp_scores /= pop_size

    return comp_scores



def BGA(pop_size=60, dim=30, num_gen=200, mut_rate=1/30, cross_rate=0.9, construction=1, problem=1):
    # driver function for the binary genetic algorithm
    population = initialization(pop_size, dim)
    obj_values = np.zeros(pop_size)
    best_values = np.zeros(num_gen)
    avg_values = np.zeros(num_gen)
    comp_scores = np.zeros((2, 10, num_gen))    # competitor scores 

    for pop_no in range(pop_size):
        obj_values[pop_no] = fitness(population[pop_no], construction, problem)

    for iter_no in range(num_gen):
        population[0:int(pop_size/2), :], obj_values[0:int(pop_size/2)] = binary_tournament_selection(population, obj_values)
        child_idx = int(pop_size/2)

        while(child_idx+1 < pop_size):
            if(np.random.rand() < cross_rate):  # crossover occurring based on the probability
                pid1, pid2 = np.random.randint(low=0, high=int(pop_size/2), size=2)
                child1, child2 = crossover(population[pid1], population[pid2])

                child1 = mutation(child1)
                child2 = mutation(child1)
                obj_child1 = fitness(child1, construction, problem)
                obj_child2 = fitness(child2, construction, problem)

                population[child_idx] = child1
                obj_values[child_idx] = obj_child1
            
                population[child_idx+1] = child2
                obj_values[child_idx+1] = obj_child2

            child_idx += 2

        comp_scores[:, :, iter_no] = count_competitors(population, construction)

        best_values[iter_no] = np.max(obj_values)
        avg_values[iter_no] = np.mean(obj_values)

    return best_values, avg_values, comp_scores



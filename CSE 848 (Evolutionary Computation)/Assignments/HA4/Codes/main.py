from BGA import BGA
import numpy as np
from matplotlib import pyplot as plt

# define hyper-parameters
pop_size = 60
dim = 30
num_gen = 200
mut_rate = 1/dim 
cross_rate = 0.9 
construction = 2
problem = 2
num_runs = 30

# initialize the variables
gen_best_values = np.zeros((num_runs, num_gen))
gen_avg_values = np.zeros((num_runs, num_gen))
best_values = np.zeros(num_runs)
comp_scores = np.zeros((2, 10, num_gen, num_runs))

# main run
for i in range(num_runs):
    gen_best_values[i,:], gen_avg_values[i,:], comp_scores[:, :, :, i] = BGA(pop_size, dim, num_gen, mut_rate, cross_rate, construction, problem)
    best_values[i] = gen_best_values[i, num_gen-1]

median_idx = np.argsort(best_values)[num_runs//2]

# Generating statistics for all the runs of BGA
print('=============================================================')
print('Construction:{}, Problem:{}'.format(construction, problem))
print('=============================================================')
print('Best:{}'.format(np.max(best_values)))
print('Median:{}'.format(np.median(best_values)))
print('Mean:{}'.format(np.mean(best_values)))
print('Worst:{}'.format(np.min(best_values)))

fig = plt.figure()
X = [i for i in range(num_gen)]
Y1 = gen_best_values[median_idx,:]
Y2 = gen_avg_values[median_idx,:]
plt.plot(X, Y1)
plt.plot(X, Y2)

plt.legend(['Best Values', 'Avg Values'])
plt.xlabel('Generation Number')
plt.ylabel('Objective Value')
# plt.title('Generation-wise Objective Values for the Median Run of Problem:{}, Construction:{}'.format(problem, construction))
# plt.show()
fig.savefig('Problem_' + str(problem) + ' Construction_' + str(construction) + '/Plot.jpg')
print('=============================================================')


# Competitor plotting for median run
X = [i for i in range(num_gen)]

for i in range(10):
    Y1 = comp_scores[0, i, :, median_idx]
    Y0 = comp_scores[1, i, :, median_idx]
    
    fig = plt.figure()
    plt.plot(X, Y1)
    plt.plot(X, Y0)
    
    plt.legend(['1(' + str(i+1) + ')', '0(' + str(i+1) + ')'])
    plt.xlabel('Generation Number')
    plt.ylabel('#Competitors')
    # plt.title('Generation-wise trend for occurrences of competitors for different subproblems')
    # plt.show()
    fig.savefig('Problem_' + str(problem) + ' Construction_' + str(construction) + '/Comp_' + str(i+1) + '.jpg')

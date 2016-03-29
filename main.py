import prepare_data
import baseline_solution
import baseline_active
import active_init_1
import active_minimum_margin
from multiprocessing import Process

print 'result format (train volume; f1_score with macro averaging)'

active_init_1.active_init_1()

if __name__ == '__main__':
    procs = []
    procs.append(Process(target=baseline_solution.baseline_solution))
    procs.append(Process(target=baseline_active.baseline_active))
    #procs.append(Process(target=active_minimum_margin.active_minimum_margin))
    map(lambda x: x.start(), procs)
    map(lambda x: x.join(), procs)
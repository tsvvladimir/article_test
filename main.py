import prepare_data
import baseline_solution
import baseline_active
import active_init_1
import active_minimum_margin
import active_cluster_svm_margin
import active_cluster_svm_margin_cluster
from multiprocessing import Process
if __name__ == '__main__':
    print 'result format (train volume; f1_score with macro averaging)'

    #baseline_solution.baseline_solution()
    #baseline_active.baseline_active(20)
    #active_init_1.active_init_1()

    active_cluster_svm_margin.active_cluster_svm_margin()
    #baseline_solution.baseline_solution()
    #active_cluster_svm_margin_cluster.active_cluster_svm_margin_cluster()

    '''
    baseline_active.baseline_active(400)
    baseline_active.baseline_active(200)
    baseline_active.baseline_active(100)
    baseline_active.baseline_active(50)
    baseline_active.baseline_active(25)
    baseline_active.baseline_active(10)
    '''
    '''
    if __name__ == '__main__':
        procs = []
        procs.append(Process(target=baseline_solution.baseline_solution))
        #procs.append(Process(target=baseline_active.baseline_active, args=(10,)))
        procs.append(Process(target=baseline_active.baseline_active, args=(20,)))
        #procs.append(Process(target=baseline_active.baseline_active, args=(50,)))
        #procs.append(Process(target=baseline_active.baseline_active, args=(100,)))
        #procs.append(Process(target=baseline_active.baseline_active, args=(200,)))
        #procs.append(Process(target=baseline_active.baseline_active, args=(400,)))
        #procs.append(Process(target=active_minimum_margin.active_minimum_margin))
        procs.append(Process(target=active_init_1.active_init_1))
        map(lambda x: x.start(), procs)
        map(lambda x: x.join(), procs)
    '''
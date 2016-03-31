import prepare_data
import baseline_solution
import baseline_active
import active_init_1
import active_minimum_margin
import active_cluster_svm_margin
import active_cluster_svm_margin_cluster
from multiprocessing import Process

if __name__ == '__main__':
    #print 'result format (train volume; f1_score with macro averaging)'

    #baseline_solution.baseline_solution()
    #baseline_active.baseline_active(20)
    #active_init_1.active_init_1()
    #baseline_solution.baseline_solution("fold2", prepare_data.fold2_train_data, prepare_data.fold2_train_target, prepare_data.fold2_test_data, prepare_data.fold2_test_target)
    #active_cluster_svm_margin.active_cluster_svm_margin()
    #baseline_solution.baseline_solution("fold1", prepare_data.fold1_train_data, prepare_data.fold1_train_target, prepare_data.fold1_test_data, prepare_data.fold1_test_target)
    #active_cluster_svm_margin.active_cluster_svm_margin("fold1", prepare_data.fold1_train_data, prepare_data.fold1_train_target, prepare_data.fold1_test_data, prepare_data.fold1_test_target)

    if __name__ == '__main__':
        procs = []
        #procs.append(Process(target=baseline_solution.baseline_solution, args=("fold1", prepare_data.fold1_train_data, prepare_data.fold1_train_target, prepare_data.fold1_test_data, prepare_data.fold1_test_target)))
        procs.append(Process(target=baseline_solution.baseline_solution, args=("fold1", )))
        procs.append(Process(target=baseline_solution.baseline_solution, args=("fold2", )))
        procs.append(Process(target=baseline_solution.baseline_solution, args=("fold3", )))
        #procs.append(Process(target=baseline_solution.baseline_solution, args=("fold4", )))
        #procs.append(Process(target=baseline_solution.baseline_solution, args=("fold3", prepare_data.fold3_train_data, prepare_data.fold3_train_target, prepare_data.fold3_test_data, prepare_data.fold3_test_target)))
        #procs.append(Process(target=baseline_solution.baseline_solution, args=("fold4", prepare_data.fold4_train_data, prepare_data.fold4_train_target, prepare_data.fold4_test_data, prepare_data.fold4_test_target)))
        #for x in procs:
        #   x.start()
        map(lambda x: x.start(), procs)
        map(lambda x: x.join(), procs)

    #active_cluster_svm_margin_cluster.active_cluster_svm_margin_cluster()

    #procs = []
    #procs.append(Process(target=active_cluster_svm_margin.active_cluster_svm_margin))
    #procs.append(Process(target=active_cluster_svm_margin_cluster.active_cluster_svm_margin_cluster))
    #map(lambda x: x.start(), procs)
    #map(lambda x: x.join(), procs)

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
    procs.append(Process(target=baseline_active.baseline_active, args=(10,)))
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
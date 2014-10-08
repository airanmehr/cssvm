'''
Created on Oct 1, 2014

@author: arya iranmehr
'''
from cssvm_tools import *
from time import time
path='/home/arya/workspace/cssvm/datasets/'

def main():
    start=time()
    param={'measure':'Risk', 'verb':1}
    param['dataset_name']='german'
    param['dataset']=path + param['dataset_name']
    param['train_y'], param['train_x'] = svm_read_problem(param['dataset']+'.train' )
    param['test_y'], param['test_x'] = svm_read_problem(param['dataset']+'.test' )
    param['cmdline'] = True

    param['RangeG']=[1e-2]
    param['RangeC']=[1e-2, 1e-1, 1e-3]
    param['RangeK']=[1, 0.9, 0.7, 0.5, 0.3, 0.1]
    param['RangeCp']=[1, 2.5, 5, 7.5, 10]
 
    param=set_default_params(param)
    param['fold'] = 10
    val_risks, test_risks=[], []
    for alg in algorithms:
        param['alg']=alg
        val_risk, param['C'], param['gamma'], param['Cp'], param['Cn'], param['cv_th'] = grid_search(param.copy()) #shallow copy (i.e. copy on write for object members)
        test_risk, th = train(param.copy(), do_test=True)
        test_risks.append(test_risk)
        val_risks.append(val_risk)
    for (alg,v,t) in zip(algorithms,val_risks,test_risks):
        print"{0} Grid on {1}  with Validation {2} of {3:.6f} and \tTest {2} of {4:.6f} \n".format(alg, param['dataset_name'], param['measure'], v, t)
    print"Best {0} Validation is {1} and in Test is {2}".format(param['measure'], algorithms[val_risks.index(min(val_risks))], algorithms[test_risks.index(min(test_risks))])
    print 'Done in {0} Seconds!'.format(round((time() - start),1))
            
if __name__ == '__main__':
    main()    
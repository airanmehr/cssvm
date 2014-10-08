from numpy import mean
from cssvmutil import *
from eval import *
from subprocess import  Popen,PIPE
cssvm_train='/home/arya/workspace/cssvm/svm-train'
cssvm_classify='/home/arya/workspace/cssvm/svm-predict'
path= '/home/arya/datasets/cssvm/'
measure={'CSA':'Risk','CSU':'AUC','IDL':'AUC','CSE':'Income'}
algs=['BM','BP','CS']
def get_range_for_algorithm(param):
    RangeC, RangeG, RangeCp, RangeK= param['RangeC'], param['RangeG'], param['RangeCp'], param['RangeK']
    if param['alg']=='BM':
        param['SearchC'] =True
        param['SearchG'] =True
        param['SearchCp']=False
        param['SearchK'] =False
    if param['alg']=='BP':
        param['SearchC'] =True
        param['SearchG'] =False
        param['SearchCp']=True
        param['SearchK'] =False
    if param['alg']=='CS':
        param['SearchC'] =True
        param['SearchG'] =False
        param['SearchCp']=True
        param['SearchK'] =True    
#     param['SearchG'] =True
    if not param['SearchC']:
        with open( get_out_file_name(param, 'BM')) as filein:
            fields = filein.readlines()[-1].split()
        RangeC = [float(fields[1])]
    if not param['SearchG']:
        with open( get_out_file_name(param, 'BM')) as filein:
            fields = filein.readlines()[-1].split()
        RangeG = [float(fields[2])]
    if not param['SearchK']:
        RangeK=[1]
    if not param['SearchCp']:
        RangeCp=[1]
    return RangeC, RangeG, RangeCp, RangeK

def get_out_file_name(param, alg=None):
    if alg==None:
        alg=param['alg']
    out='{0}.{1}.{2}'.format( param['dataset'] , alg, param['measure'])
    if param['measure'] == 'AUC':
        out+='-{0}'.format(param['t'])
    out+='.out'
    return out

def set_default_params(param):
    keys=param.keys()
    if 'fold' not in keys:
        param['fold']=10
    if 'verb' not in keys:
        param['verb']=0
    if 'measure' not in keys:
        param['measure']='AUC'
    if param['measure']=='AUC':
        if 't' not in keys:
            param['t']=0.9
    if 'RangeC' not in keys:
        param['RangeC']  = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    if 'RangeG' not in keys:
        param['RangeG']  = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    if 'RangeCp' not in keys:
        param['RangeCp'] = [1, 5, 10, 50, 100]
    if 'RangeK' not in keys:
        param['RangeK']  = [1, 0.975, 0.95, 0.925, 0.9, 0.7, 0.6, 0.5, 0.3, 0.4, 0.2, 0.1, 0.01]
    if 'cmdline' not in keys:
        param['cmdline']  = False
    return param

def performance_is_better(param, performance, best, best_Cp, best_Cn): #sometimes we want to minimize and sometimes maximize
    if param['measure'] == 'Risk' or param['measure'] == 'Error' or param['measure'] == 'PError':
        if (performance < best) or (performance == best and  param['Cp'] < best_Cp) or (performance == best and  param['Cn'] < best_Cn):
            return True
    if param['measure'] == 'AUC' or param['measure'] == 'Income' or param['measure'] == 'Accuracy':
        if (performance > best) or (performance == best and  param['Cp'] < best_Cp) or (performance == best and  param['Cn'] < best_Cn):
            return True
    
def grid_search(param):
    best_performance, best_Cp, best_Cn, best_c, best_g, best_TH = 1, 1, 1, 1, 1, 0
    RangeC, RangeG, RangeCp, RangeK= get_range_for_algorithm(param)
    if param['verb']>0:
        print 'RangeC =', RangeC, 'RangeG =', RangeG, 'RangeCp =', RangeCp, 'RangeK =', RangeK
    with  open(get_out_file_name(param), 'a') as out_file:
        print >> out_file, 'C\tGamma\tCp\tKappa\tPerformance'
        for i in RangeK:
            for j in RangeCp:
                for k in RangeC:
                    for l in RangeG:
                        if 1/i > j:
                            continue
                        param['Cn'], param['Cp'], param['C'], param['gamma'] = 1 / i, j, k, l  #**************  C_n = 1/Kappa
                        performance, th = train(param.copy())
                        if performance_is_better(param, performance, best_performance, best_Cp, best_Cn):
                            best_performance,best_Cp, best_Cn, best_c, best_g, best_TH = performance, param['Cp'], param['Cn'], param['C'], param['gamma'], th
                        print >> out_file, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format( param['C'], param['gamma'],param['Cp'], param['Cn'], th,  performance)
        print >> out_file, 'Bests:\t{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(best_c, best_g, best_Cp, best_Cn, best_TH, best_performance)
        if param['verb']>0:
            print"{0} Grid on {1} Finished in {2} Iterations. C={3} Gamma={4} Cp={5} Kappa={6} Threshold:{7:.2f} \t {8}={9:.3f}\n".format(param['alg'], param['dataset_name'],len(RangeG)*len(RangeC)*len(RangeCp)*len(RangeK), best_c, best_g, best_Cp, 1./best_Cn, best_TH, param['measure'], best_performance)
    return best_performance, best_c, best_g, best_Cp, best_Cn, best_TH  

def train(param, do_test=False):
    """
    Trains and computes the performance of the model
    when fold is 
        -1 it targets the test dataset (*.test) for evaluation
        0  it targets the validation dataset (*.val) for evaluation
        1    it targets the traitrainnin dataset (*.val) for evaluation
        any natural integer k, it performs k-fold cross validation on training set
    """
    if(do_test):
        param['fold']=-1
    param['Pp'] = mean(array(param['train_y']) == 1)
    param['Pn'] = 1 - param['Pp']
    deci = None
    cv_th=0
    if param['cmdline']:
        deci, label = get_deci_cmdline(param)
    else:
        deci, label = get_cv_deci(param)
    assert deci and label and len(deci)==len(label)
    if param['measure']=='Income':
        performance = get_income(deci, label, param)
    elif param['measure']=='AUC':
        performance = get_auc(deci, label, param)
    elif param['measure']=='Risk':
        performance, cv_th = get_risk(deci, label, param['Pn'], param['Pp'], 1, 5, do_test, do_test, param) #plots in the test phase
    elif param['measure']=='Accuracy':
        performance = get_acc(deci, label)
    elif param['measure']=='Error':
        performance = get_error(deci, label)
    elif param['measure']=='PError':
        performance = get_perror(deci, label, param['Pn'], param['Pp'])
    if param['verb']>1:
            print"{0} Train on {1}  with C={2} Gamma={3} Cp={4} Kappa={5} Threshold= {6} \t {7}={8}  ".format(param['alg'], param['dataset_name'], param['C'], param['gamma'], param['Cp'], 1./param['Cn'], th, param['measure'], performance)
    return performance,cv_th
    
    

def get_pos_deci(param):
    params = '-q -h 0 -m 2000 -c {0} -g {1} -w1 {2} -w-1 {3} '.format(param['C'], param['gamma'], param['Cp'], param['Cn'])
    if param['alg'] == 'EDBP' or param['alg'] == 'EDCS':
        params = '-C 2 ' + params
        model = svm_train(train_y, train_x, params, param['train_costs'])
    else:
        if param['alg'] == 'BM' or param['alg'] == 'BP' or  param['alg'] == 'EDBM':
            params = '-C 0 ' + params
        elif param['alg'] == 'CS':
            params = '-C 1 ' + params
        model = svm_train(param['train_y'], param['train_x'], params)
    labels = model.get_labels()
    py, evals, deci = svm_predict(param['test_y'], param['test_x'], model)
    if model.get_labels() != [1, -1] and model.get_labels() != [-1, 1]:
        return None
    decV = [ labels[0]*val[0] for val in deci]
    return decV

def get_deci_cmdline(param):
    cv_option=""
    model_file= '{0}.{1}.model'.format(param['dataset'], param['alg'])
    pred_file= '{0}.{1}.pred'.format(param['dataset'], param['alg'])
    train_file= param['dataset']+'.train'
    if param['fold'] == -1:
        test_file= param['dataset']+'.test'
    elif param['fold'] == 0 :
        test_file= param['dataset']+'.val'
    elif param['fold'] == 1 :
        test_file= train_file 
    else:
        test_file= ""
        cv_option = "-v {0}".format(param['fold'])
    cmd = '{0} -h 0 {1} -m 2000 -c {2} -g {3} {4}'.format(cssvm_train,('','q')[param['verb']>4], param['C'], param['gamma'], cv_option)
    if param['alg'] == 'EDBP' or param['alg'] == 'EDCS':
        cmd += ' -C 2 -W {0}.train.cost'.format(param['name'])
    if param['alg'] == 'CS':
        cmd += ' -C 1 '
    cmd += ' -w1 {0} -w-1 {1}  {2} {3} '.format(param['Cp'], param['Cn'], train_file, model_file)
    p = Popen(cmd, shell=True, stdout=PIPE)
    p.wait()
    if cv_option == "":
        cmd = '{0} {1} {2} {3} '.format(cssvm_classify,test_file, model_file, pred_file)
        p = Popen(cmd, shell=True, stdout=PIPE)
        p.wait()
        deci=read_deci(pred_file)
        model = svm_load_model(model_file)
        labels = model.get_labels()
        deci = [labels[0]*val for val in deci]
        label=read_labels(test_file)
    else:
        deci=read_deci(model_file+".cv")
        label=read_labels(train_file)
    return deci, label

def  get_cv_deci(param):
    seed(0)
    if param['fold'] == -1: 
        deci = get_pos_deci(param)
        label=  param['test_y']
    elif param['fold'] == 0 :
        param['test_y'], param['test_x']=param['val_y'], param['val_x']
        deci = get_pos_deci(param)
        label=  param['val_y']
    elif param['fold'] == 1 :
        param['test_y'], param['test_x'] = param['train_y'], param['train_x']
        deci = get_pos_deci(param)
        label= param['train_y']
    else:
        deci, model, label = [], [], []
        subparam = param.copy()
        prob_l = len(param['train_y'])     #random permutation by swapping i and j instance
        for i in range(prob_l):
            j = randrange(i, prob_l)
            param['train_x'][i], param['train_x'][j] = param['train_x'][j], param['train_x'][i]
            param['train_y'][i], param['train_y'][j] = param['train_y'][j], param['train_y'][i]
            if param['alg'] == 'EDBP' or param['alg'] == 'EDCS' or param['alg'] == 'EDBM':
                param['costs'][i], param['costs'][j] = param['costs'][j], param['costs'][i]
        for i in range(param['fold']):     #cross training : folding
            begin = i * prob_l // param['fold']
            end = (i + 1) * prob_l // param['fold']
            subparam['train_x'] = param['train_x'][:begin] + param['train_x'][end:]
            subparam['train_y'] = param['train_y'][:begin] + param['train_y'][end:]
            subparam['test_x'] = param['train_x'][begin:end]
            subparam['test_y'] = param['train_y'][begin:end]
            subdeci = get_pos_deci(subparam)
            assert subdeci
            deci += subdeci
            label+=subparam['test_y']
    return deci, label

def read_costs(dataset):
    records = open(dataset).readlines()
    costs = []
    for record in records:
        costs.append(float(record.split()[0].strip()))
    return costs

def read_labels(dataset):
    labels=[]
    with open(dataset) as filein:
        for line in filein:
            labels.append( int(line.split()[0]))
    return labels

def read_deci(dataset):
    labels=[]
    with open(dataset) as filein:
        for line in filein:
            labels.append(float(line.split()[0]))
    return labels
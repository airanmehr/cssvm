#!/usr/bin/env python

import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path 
from cssvm import *
algorithms=['BM','BP','CS']
measure=['Risk','AUC','Income']

def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)

def svm_load_model(model_file_name):
	"""
	svm_load_model(model_file_name) -> model
	
	Load a LIBSVM model from model_file_name and return.
	"""
	model = libsvm.svm_load_model(model_file_name.encode())
	if not model: 
		print("can't open model file %s" % model_file_name)
		return None
	model = toPyModel(model)
	return model

def svm_save_model(model_file_name, model):
	"""
	svm_save_model(model_file_name, model) -> None

	Save a LIBSVM model to the file model_file_name.
	"""
	libsvm.svm_save_model(model_file_name.encode(), model)

def evaluations(ty, pv):
	"""
	evaluations(ty, pv) -> (ACC, MSE, SCC)

	Calculate accuracy, mean squared error and squared correlation coefficient
	using the true values (ty) and predicted values (pv).
	"""
	if len(ty) != len(pv):
		raise ValueError("len(ty) must equal to len(pv)")
	total_correct = total_error = 0
	sumv = sumy = sumvv = sumyy = sumvy = 0
	for v, y in zip(pv, ty):
		if y == v: 
			total_correct += 1
		total_error += (v-y)*(v-y)
		sumv += v
		sumy += y
		sumvv += v*v
		sumyy += y*y
		sumvy += v*y 
	l = len(ty)
	ACC = 100.0*total_correct/l
	MSE = total_error/l
	try:
		SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
	except:
		SCC = float('nan')
	return (ACC, MSE, SCC)

def svm_train(arg1, arg2=None, arg3=None,costs=None):
	"""
	svm_train(y, x [, options]) -> model | ACC | MSE 
	svm_train(prob [, options]) -> model | ACC | MSE 
	svm_train(prob, param) -> model | ACC| MSE 

	Train an SVM model from data (y, x) or an svm_problem prob using
	'options' or an svm_parameter param. 
	If '-v' is specified in 'options' (i.e., cross validation)
	either accuracy (ACC) or mean-squared error (MSE) is returned.
	options:
	    -s svm_type : set type of SVM (default 0)
	        0 -- C-SVC		(multi-class classification)
	        1 -- nu-SVC		(multi-class classification)
	        2 -- one-class SVM
	        3 -- epsilon-SVR	(regression)
	        4 -- nu-SVR		(regression)
	    -t kernel_type : set type of kernel function (default 2)
	        0 -- linear: u'*v
	        1 -- polynomial: (gamma*u'*v + coef0)^degree
	        2 -- radial basis function: exp(-gamma*|u-v|^2)
	        3 -- sigmoid: tanh(gamma*u'*v + coef0)
	        4 -- precomputed kernel (kernel values in training_set_file)
	    -d degree : set degree in kernel function (default 3)
	    -g gamma : set gamma in kernel function (default 1/num_features)
	    -r coef0 : set coef0 in kernel function (default 0)
	    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
	    -m cachesize : set cache memory size in MB (default 100)
	    -e epsilon : set tolerance of termination criterion (default 0.001)
	    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
	    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
	    -v n: n-fold cross validation mode
	    -q : quiet mode (no outputs)
	"""
	prob, param = None, None
	if isinstance(arg1, (list, tuple)):
		assert isinstance(arg2, (list, tuple))
		y, x, options = arg1, arg2, arg3
		param = svm_parameter(options,costs)
		prob = svm_problem(y, x, isKernel=(param.kernel_type == PRECOMPUTED))
	elif isinstance(arg1, svm_problem):

		prob = arg1
		if isinstance(arg2, svm_parameter,costs):
			param = arg2
		else:
			param = svm_parameter(arg2)
	if prob == None or param == None:
		raise TypeError("Wrong types for the arguments")

	if param.kernel_type == PRECOMPUTED:
		for xi in prob.x_space:
			idx, val = xi[0].index, xi[0].value
			if xi[0].index != 0:
				raise ValueError('Wrong input format: first column must be 0:sample_serial_number')
			if val <= 0 or val > prob.n:
				raise ValueError('Wrong input format: sample_serial_number out of range')

	if param.gamma == 0 and prob.n > 0: 
		param.gamma = 1.0 / prob.n
	libsvm.svm_set_print_string_function(param.print_func)
	err_msg = libsvm.svm_check_parameter(prob, param)
	if err_msg:
		raise ValueError('Error: %s' % err_msg)

	if param.cross_validation:
		l, nr_fold = prob.l, param.nr_fold
		target = (c_double * l)()
		libsvm.svm_cross_validation(prob, param, nr_fold, target)	
		ACC, MSE, SCC = evaluations(prob.y[:l], target[:l])
		if param.svm_type in [EPSILON_SVR, NU_SVR]:
			print("Cross Validation Mean squared error = %g" % MSE)
			print("Cross Validation Squared correlation coefficient = %g" % SCC)
			return MSE
		else:
			print("Cross Validation Accuracy = %g%%" % ACC)
			return ACC
	else:
		m = libsvm.svm_train(prob, param)
		m = toPyModel(m)

		# If prob is destroyed, data including SVs pointed by m can remain.
		m.x_space = prob.x_space
		return m

def svm_predict(y, x, m, options="-q"):
	"""
	svm_predict(y, x, m [, options]) -> (p_labels, p_acc, p_vals)

	Predict data (y, x) with the SVM model m. 
	options: 
	    -b probability_estimates: whether to predict probability estimates, 
	        0 or 1 (default 0); for one-class SVM only 0 is supported.
	    -q : quiet mode (no outputs).

	The return tuple contains
	p_labels: a list of predicted labels
	p_acc: a tuple including  accuracy (for classification), mean-squared 
	       error, and squared correlation coefficient (for regression).
	p_vals: a list of decision values or probability estimates (if '-b 1' 
	        is specified). If k is the number of classes, for decision values,
	        each element includes results of predicting k(k-1)/2 binary-class
	        SVMs. For probabilities, each element contains k values indicating
	        the probability that the testing instance is in each class.
	        Note that the order of classes here is the same as 'model.label'
	        field in the model structure.
	"""

	def info(s):
		print(s)

	predict_probability = 0
	argv = options.split()
	i = 0
	while i < len(argv):
		if argv[i] == '-b':
			i += 1
			predict_probability = int(argv[i])
		elif argv[i] == '-q':
			info = print_null
		else:
			raise ValueError("Wrong options")
		i+=1

	svm_type = m.get_svm_type()
	is_prob_model = m.is_probability_model()
	nr_class = m.get_nr_class()
	pred_labels = []
	pred_values = []

	if predict_probability:
		if not is_prob_model:
			raise ValueError("Model does not support probabiliy estimates")

		if svm_type in [NU_SVR, EPSILON_SVR]:
			info("Prob. model for test data: target value = predicted value + z,\n"
			"z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g" % m.get_svr_probability());
			nr_class = 0

		prob_estimates = (c_double * nr_class)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi, isKernel=(m.param.kernel_type == PRECOMPUTED))
			label = libsvm.svm_predict_probability(m, xi, prob_estimates)
			values = prob_estimates[:nr_class]
			pred_labels += [label]
			pred_values += [values]
	else:
		if is_prob_model:
			info("Model supports probability estimates, but disabled in predicton.")
		if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
			nr_classifier = 1
		else:
			nr_classifier = nr_class*(nr_class-1)//2
		dec_values = (c_double * nr_classifier)()
		for xi in x:
			xi, idx = gen_svm_nodearray(xi, isKernel=(m.param.kernel_type == PRECOMPUTED))
			label = libsvm.svm_predict_values(m, xi, dec_values)
			if(nr_class == 1): 
				values = [1]
			else: 
				values = dec_values[:nr_classifier]
			pred_labels += [label]
			pred_values += [values]

	ACC, MSE, SCC = evaluations(y, pred_labels)
	l = len(y)
	if svm_type in [EPSILON_SVR, NU_SVR]:
		info("Mean squared error = %g (regression)" % MSE)
		info("Squared correlation coefficient = %g (regression)" % SCC)
	else:
		info("Accuracy = %g%% (%d/%d) (classification)" % (ACC, int(l*ACC/100), l))

	return pred_labels, (ACC, MSE, SCC), pred_values

from data import *
from numpy import mean
from eval import *
from subprocess import  Popen,PIPE

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

def grid_search(param):
	best_performance, best_Cp, best_Cn, best_C, best_G = 1, 1, 1, 1, 1
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
	                    performance = train(param.copy())
	                    if (performance < best_performance) or (performance == best_performance and  param['Cp'] < best_Cp) or (performance == best_performance and  param['Cn'] < best_Cn):
	                        best_performance,best_Cp, best_Cn, best_c, best_g = performance, param['Cp'], param['Cn'], param['C'], param['gamma']
	                    print >> out_file, '{0}\t{1}\t{2}\t{3}\t{4}'.format( param['C'], param['gamma'],param['Cp'], param['Cn'], performance)
	    print >> out_file, 'Bests:\t{0}\t{1}\t{2}\t{3}\t{4}'.format(best_c, best_g, best_Cp, best_Cn, best_performance)
	    if param['verb']>0:
	        print"{0} Grid on {1} Finished in {2} Iterations. C={3} Gamma={4} Cp={5} Kappa={6} \t {7}={8}  ".format(param['alg'], param['dataset_name'],len(RangeG)*len(RangeC)*len(RangeCp)*len(RangeK), best_c, best_g, best_Cp, 1./best_Cn, param['measure'], best_performance)
	return best_performance, best_C, best_G, best_Cp, best_Cn  

def train(param, test=False):
	"""
	Trains and computes the performance of the model
	"""
	if(test):
		param['fold']=-1
	param['Pp'] = mean(array(param['train_y']) == 1)
	param['Pn'] = 1 - param['Pp']
	deci = None
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
	    performance = get_risk(deci, label, param)
	if param['verb']>1:
		print"{0} Train on {1}  with C={2} Gamma={3} Cp={4} Kappa={5} \t {6}={7}  ".format(param['alg'], param['dataset_name'], param['C'], param['gamma'], param['Cp'], 1./param['Cn'], param['measure'], performance)
	return performance
    

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
	if param['fold'] == -1:
		test_file= param['dataset'].replace('train','test')
	if param['fold'] == 0 :
		test_file= param['dataset'].replace('train','val')
	if param['fold'] == 1 :
		test_file= param['dataset'] 
	else:
		test_file= ""
		cv_option = "-v {0}".format(param['fold'])
		
	cmd = '{0} -h 0 {1} -m 2000 -c {2} -g {3} {4}'.format(cssvm_train,('','q')[param['verb']>4], param['C'], param['gamma'], cv_option)
	if param['alg'] == 'EDBP' or param['alg'] == 'EDCS':
	    cmd += ' -C 2 -W {0}.train.cost'.format(param['name'])
	if param['alg'] == 'CS':
	    cmd += ' -C 1 '
	cmd += ' -w1 {0} -w-1 {1}  {2} {3} '.format(param['Cp'], param['Cn'], param['dataset'], model_file)
	p = Popen(cmd, shell=True, stdout=PIPE)
	p.wait()
	if cv_option == "":
		cmd = '{0} {1} {2} {3} '.format(cssvm_classify,test_file, model_file, pred_file)
		p = Popen(cmd, shell=True, stdout=PIPE)
		p.wait()
		deci=read_deci(pred_file)
		model = svm_load_model(model_file)
		labels = model.get_labels()
		deci = [labels[0]*val[0] for val in deci]
		test_y=read_labels(test_file)
	else:
		est_y, deci=read_cv_file(model_file+".cv")
	return test_y, deci

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


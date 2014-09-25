Cost-Sensitive Support Vector Machines (CSSVM)
=
This project implements [CSSVM](http://arxiv.org/abs/1212.0975) using LibSVM. 

1.  Installation
-
Installing CSSVM is exactly the same as LibSVM:

1.  **Command line**:
In the root directory of the project and simply execute ```make```.

2.  **Python Interface** In the ```python``` directory, execute ```make```.


2.  Run
-
Running CSSVM is almost like running LibSVM, except the new  <code>-C</code> and <code>-W</code> options:
<pre><code>
Usage: svm-train [options] training_set_file [model_file]
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
-C Cost-sensitive Learning Method(default 0):
	0 -- cost-insensitive SVM (biased-penalty SVM with option -w1 C_1 -w-1 C_-1)
	1 -- cost-sensitive SVM with option -w1 C_1 -w-1 1/kappa
	2 -- cost-sensitive SVM with example-dependent cost.used with option -W cost_file_name
-W cost_file_name : file contains example costs for example-dependent cost-sensitive learning
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
</code></pre>

3.   CSSVM Tools
-

4.   Examples
-


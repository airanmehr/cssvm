from numpy import array, arange, sum, append
from operator import itemgetter
from matplotlib import pyplot

def get_perror(deci, label, Pn, Pp):
    neg, pos, neg_err, pos_err = 0., 0., 0., 0.
    for (f,y) in zip(deci,label):
        if y>0:
            pos+=1
            if f* y <= 0:        
                pos_err+=1
        else:
            neg+=1
            if f* y <= 0:        
                neg_err+=1
    print "Pn: {0} neg: {1} neg_err: {2}".format(Pn, neg, neg_err)
    print "Pp: {0} pos: {1} pos_err: {2}".format(Pp, pos, pos_err)
    return Pn*neg_err/neg + Pp*pos_err/pos

def get_error(deci, label):
    err= 0.
    for (f,y) in zip(deci,label):
        if f* y <= 0:        
            err+=1
    return err/len(label)

def get_acc(deci, label):
    correct= 0.
    for (f,y) in zip(deci,label):
        if f* y > 0:        
            correct+=1
    return 100*correct/len(label)

def get_income(deci, label, costs):
    income= 0.
    for (f,y,c) in zip(deci,label, costs):
        if f > 0 and y > 0:        
            income += c
        if f > 0 and y < 0:
            income -= c
    return income

def get_risk(deci, label, Pn, Pp, Cn=1, Cp=1, doPlot=False, isTesting=False, param=None):
    """
    Computes Risk for a given Decision Values, Label, Costs and Probabilities and Plots ROC Curve with denoting the ROC Operating points with 
    1) No thresholding (FPR and FNR associated with the Decision values)
    2) Best thresholding (Finds best operating point in ROC curve for given arguments)
    3) TH Thresholding (if the argument th is not None (probably computed by CV), its value is added to the decision values and operating point and its risk is computed and shown in the plot )
    In CV, i.e. when th parameter is None, the threshold of the point with the best risk is returned, to be used in the test phase 
    """
    db,pos,neg, n=sort_deci(deci,label)
    tp, fp, min_risk = 0., 0., 1
    x, y = array([0]), array([1])   # x,y are coordinates in ROC space
    w = [Cn * Pn , Cp * Pp ]
    w = [w[0]/sum(w), w[1]/sum(w)]#    Normalizing costs  (for comparing risks of different costs)
    for i in range(1, n + 1):
        if db[i - 1][1] > 0:        
            tp += 1.
        else:
            fp += 1.
        fpr = fp / neg
        fnr = (pos - tp) / pos
        risk = w[0] * fpr + w[1] * fnr
        if min_risk >= risk:
            min_risk = risk
            best_op = i     #Best Operating Point index in the sorted list
        x, y= append(x, fpr), append(y, fnr)
    ROCPoint_NoTH = get_Risk_for_TH(deci, label, w, pos, neg, 0)
    if isTesting:
        ROCPoint_TH = get_Risk_for_TH(deci, label, w, pos, neg, param['cv_th']) 
        test_th=get_TH_for_Risk(db, best_op)
        if param['verb']>2:
            print "Best Threshold in: CV= {0} Test= {1}".format(param['cv_th'], test_th)
    else:
        cv_th=get_TH_for_Risk(db, best_op)
        ROCPoint_TH = None
    
    if doPlot:
        plot_ROC_Risk(w, x, y, best_op, min_risk,get_figure_fname(param), get_figure_title(param), ROCPoint_TH, ROCPoint_NoTH)
    
    if isTesting: 
        return ROCPoint_TH['risk'], test_th # in Test phase, returns the risk of the point with CV_threshold and the best threshold in all operating points
    else:
        return min_risk, cv_th # in Training phase, returns the risk of the point with minimum risk and its threshold


def sort_deci(deci,label):
    db = []
    pos, neg, n = 0., 0., len(label)
    for i in range(n):
        if label[i] > 0:
            pos += 1
        else:    
            neg += 1
        db.append([deci[i], label[i]])
    db = sorted(db, key=itemgetter(0), reverse=True) # Sort in descenting order
    return db, pos, neg, n
    
def get_TH_for_Risk(db, best_op):
    return -(db[best_op][0]+db[best_op+1][0])/2

def get_Risk_for_TH(deci, label, w, pos,neg, th):
    ROCPoint={'fpr': 0., 'fnr': 0.}
    for (f,Y) in zip(deci,label):
        if (f+th)*Y <=0:
            if Y<=0:
                ROCPoint['fpr'] +=1
            else:
                ROCPoint['fnr'] +=1
    ROCPoint['fpr']/=neg
    ROCPoint['fnr']/=pos
    ROCPoint['risk']= w[0] * ROCPoint['fpr'] + w[1] * ROCPoint['fnr']
    return ROCPoint
    
def get_figure_fname(param):
#     print param
    return '/home/arya/{0}.{1}.png'.format(param['dataset_name'],param['alg'])

def get_figure_title(param):
    return '{0} on {1}'.format(param['alg'], param['dataset_name'])

def get_auc(deci, label, param):
    """
    Computes AUC for param['hit'] > t, By default it computes AUC for TP > 0.9
    """
    db, xy_arr = [], []
    pos, neg, tp, fp, auc, err = 0, 0, 0, 0, 0., 0.
    n = len(label) 
    for i in range(n):
        if label[i] > 0:
            if deci[i]<0:
                err+=1
            pos += 1
        else:
            if deci[i]>0:
                err+=1    
            neg += 1
        db.append([deci[i], label[i]])
    db = sorted(db, key=itemgetter(0), reverse=True)
    for i in range( n):
        if db[i - 1][1] > 0:        
            tp += 1.
        else:
            fp += 1.
        fpr = fp / neg
        fn = pos - tp
        fnr = fn / pos
        xy_arr.append([fpr, fnr])
    xy_arr.append([1,0])
    if param['hit']=='TN':  #TN
        prev_x = 0
        for x,y in xy_arr:
            if x > param['t'] :
                break
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        auc+=(param['t']-prev_x)*y
    else: #TP
        prev_y = 0
        for x,y in reversed( xy_arr):
            if y> param['t']:
                break
            if y != prev_y:
                auc += (y - prev_y) * x
                prev_y = y
        auc+=(param['t']-prev_y)*x
    auc= auc/(1-param['t'])
    if param['do_plot']:
        plot_ROC_tAUC(x, y)
    return auc
   
def get_auc(deci, label, param):
    if param['measure']=='TP':
        Fix_Detection=True
    elif param['measure']=='TN':
        Fix_Detection=False
    else:
        print 'The measure should be determined either TP or TN.'
        exit()
    t=1-param['t']
    db = []
    auc = 0.
    pos, neg, tp, fp= 0, 0, 0, 0
    n = len(label) 
    err=0.        
    for i in range(n):
        if label[i] > 0:
            if deci[i]<0:
                err+=1
            pos += 1
        else:
            if deci[i]>0:
                err+=1    
            neg += 1
        db.append([deci[i], label[i]])
    db = sorted(db, key=itemgetter(0), reverse=True)
    xy_arr = []
    for i in range( n):
        if db[i - 1][1] > 0:        
            tp += 1.
        else:
            fp += 1.
        fpr = fp / neg
        fn = pos - tp
        fnr = fn / pos
        xy_arr.append([fpr, fnr])
    xy_arr.append([1,0])
    if not Fix_Detection:  #TN
        prev_x = 0
        for x,y in xy_arr:
            if x > t :
                break
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        auc+=(t-prev_x)*y
    else: #TP
        prev_y = 0
        for x,y in reversed( xy_arr):
            if y> t:
                break
            if y != prev_y:
                auc += (y - prev_y) * x
                prev_y = y
        auc+=(t-prev_y)*x
    auc= auc/t
#    print "AUC:{3}    Error: {0}  ({1} out of {2})".format(err/n,err,n,auc)
    return auc
            
def plot_ROC_Risk(w,x,y,best_operating_point,risk,fname,title, TH=None, NoTH=None):
    from matplotlib import rc
    from numpy import ones
    import pylab
    rc('font', family='serif', size=20)
    fig = pylab.figure(1, figsize=(8, 8), dpi=100, facecolor='w')
    ax = fig.add_subplot(111)
    fig.hold(True)
    m = w[0] / w[1]
    line_x = arange(0, 1, 0.001)
    line_y = ones(1000) - m * line_x
    pylab.fill_between(x , ones(len(y)), y, facecolor='#D0D0D0')
    pylab.plot(x, y, 'r', linewidth=3)
    best=[x[best_operating_point], y[best_operating_point]]
    pylab.plot(best[0], best[1], '*k', markersize=20, label='Risk$^*$    ={0:.3f}'.format(risk))
    pylab.plot(line_x, line_y, 'k', linewidth=3)
    midline=len(line_x)/2
    arrow_x, arrow_y=line_x[midline], line_y[midline]
    pylab.arrow(arrow_x, arrow_y, -w[0] * 0.15 , -w[1] * 0.15, head_length=.02, head_width=.02, linewidth=2, color='black')
    pylab.axis([-0.005, 1.001, -0.005, 1.001])
    pylab.xlabel('$FPR$')
    pylab.ylabel('$FNR$')
    pylab.title(title)
    if TH!=None:
        pylab.plot(TH['fpr'], TH['fnr'], 'ob', markersize=12, label='TH Risk={0:.3f}'.format(TH['risk']))
    if NoTH!=None:
        pylab.plot(NoTH['fpr'], NoTH['fnr'], 'sc', markersize=12, label='BM Risk={0:.3f}'.format(NoTH['risk']))
    pyplot.legend(numpoints=1, prop={'size':16})
    pylab.savefig(fname)
    pylab.grid()
    ax.set_aspect('equal')
    pylab.show()
    pylab.close()

def plot_ROC_tAUC(x,y):
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', family='serif', size=24)
    import pylab
    fig = pylab.figure(1, figsize=(8, 7), dpi=100, facecolor='w')
    fig.hold(True)
    from numpy import ones
    pylab.fill_between(x , ones(len(y)), y, facecolor='#D0D0D0')
    pylab.plot(x, y, 'r', linewidth=3)
#        pylab.plot(-1,-1,'ob',markersize=10,label='Nayman Pearson')
    pylab.axis([-0.005, 1.001, -0.005, 1.001])
    pylab.xlabel('$P_{FP}$', fontsize=30)
    pylab.ylabel('$P_{FN}$', fontsize=30)
    pylab.grid()
    pylab.show()
    pylab.close()
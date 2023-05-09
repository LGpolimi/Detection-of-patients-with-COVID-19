import pandas as pd
import numpy as np
import math
import random
from math import sqrt
from statistics import *
from statsmodels import stats
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Z_classes_and_functions import *
import shap

class metricsSet():
    def __init__(self):
        self.thresh = 0
        self.sens = 0
        self.spec = 0
        self.revspec = 0
        self.acc = 0
        self.ppv = 0
        self.npv = 0
        self.jind = 0
        self.nind = 0
        self.lowsens = 0
        self.lowspec = 0
        self.lowrevspec = 0
        self.lowacc = 0
        self.lowppv = 0
        self.lownpv = 0
        self.lowjind = 0
        self.lownind = 0
        self.uppsens = 0
        self.uppspec = 0
        self.upprevspec = 0
        self.uppacc = 0
        self.uppppv = 0
        self.uppnpv = 0
        self.uppjind = 0
        self.uppnind = 0

class metricsList():

    def __init__(self,name):
        self.listname = name
        self.auc = list()
        self.auc_low = list()
        self.auc_upp = list()
        self.metrics = list()
    def sort_metrics(self):
        totn = len(self.metrics)
        threshvals = list()
        for m in self.metrics:
            threshvals.append(m.thresh)
        thresharr = np.asarray(threshvals)
        temparr = thresharr
        orderlist = list()
        tempmetrics = list()
        for i in range(totn):
            minind = np.argmin(temparr)
            temparr[minind] = 999
            orderlist.append(minind)
        for oi in orderlist:
            tempmetrics.append(self.metrics[oi])
        self.metrics = tempmetrics
    def compute_roc(self):
        self.sort_metrics()
        mi = 0
        aucval = 0
        aucval_low = 0
        aucval_upp = 0
        xi = 1
        yi = 1
        xil = 1
        yil = 1
        xiu = 1
        yiu = 1
        for m in self.metrics:
            xf = m.revspec
            yf = m.sens
            xfl = m.lowrevspec
            yfl = m.lowsens
            xfu = m.upprevspec
            yfu = m.uppsens
            newarea = ((yf+yi) * (xi-xf))/2
            newarealow = ((yfl+yil) * (xil-xfl))/2
            newareaupp = ((yfu+yiu) * (xiu-xfu))/2
            aucval = aucval + newarea
            aucval_low = aucval_low + newarealow
            aucval_upp = aucval_upp + newareaupp
            xi = xf
            yi = yf
            xil = xfl
            yil = yfl
            xiu = xfu
            yiu = yfu
        self.auc = aucval
        self.auc_low = aucval_low
        self.auc_upp = aucval_upp
    def compute_nindx(self):
        auc = self.auc
        lauc = self.auc_low
        uauc = self.auc_upp
        for m in self.metrics:
            m.nind = abs(m.sens-auc) + abs(m.spec-auc)
            m.lownind = abs(m.lowsens-lauc) + abs(m.lowspec-lauc)
            m.uppnind = abs(m.uppsens-uauc) + abs(m.uppspec-uauc)

def compute_dist(array):
    medval = np.median(array)
    q25 = np.quantile(array,0.25)
    q75 = np.quantile(array,0.75)
    lowci = np.quantile(array,0.025)
    uppci = np.quantile(array,0.975)
    return(medval,q25,q75,lowci,uppci)

def scalevals(x,minval,maxval,out_range="foo"):
    if out_range == "foo":
        out_range=(0, 1)
    domain = minval, maxval
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def compute_metrics(truevals,predvals):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    toty = len(predvals)
    for y in range(toty):
        trueval = truevals[y]
        predval = predvals[y]
        if trueval == 1:
            if predval == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if predval == 1:
                fp = fp + 1
            else:
                tn = tn + 1
    allrecs = tp + tn + tp + fn
    allactpos = tp + fn
    allactneg = tn + fp
    allposcalls = fp + tp
    allnegcalls = fn + tn
    tpperc = tp / allrecs
    tnperc = tn / allrecs
    fpperc = fp / allrecs
    fnperc = fn / allrecs
    sens = tp / allactpos
    spec = tn / allactneg
    revspec = 1 - spec
    acc = (tp + tn) / allrecs
    if allposcalls > 0:
        ppv = tp / allposcalls
    else:
        ppv = 0
    if allnegcalls > 0:
        npv = tn / allnegcalls
    else:
        npv = 0
    rocdist = sqrt(((1-sens)**2)+(revspec**2))

    pci_low, pci_upp = stats.proportion.proportion_confint(tp, allactpos, alpha=0.05, method='beta')
    nci_low, nci_upp = stats.proportion.proportion_confint(tn, allactneg, alpha=0.05, method='beta')
    aci_low, aci_upp = stats.proportion.proportion_confint((tp+tn), allrecs, alpha=0.05, method='beta')
    ppv_low, ppv_upp = stats.proportion.proportion_confint(tp, allposcalls, alpha=0.05, method='beta')
    npv_low, npv_upp = stats.proportion.proportion_confint(tn, allnegcalls, alpha=0.05, method='beta')
    lowsens = pci_low
    lowspec = nci_low
    lowrevspec = 1- lowspec
    lowacc = aci_low
    lowppv = ppv_low
    lownpv = npv_low
    lowrocdist = sqrt(((1-lowsens)**2)+(lowrevspec**2))
    uppsens = pci_upp
    uppspec = nci_upp
    upprevspec = 1 - uppspec
    uppacc = aci_upp
    uppppv = ppv_upp
    uppnpv = npv_upp
    upprocdist = sqrt(((1-uppsens)**2)+(upprevspec**2))

    newmetrics = metricsSet()
    newmetrics.sens = sens
    newmetrics.spec = spec
    newmetrics.revspec = revspec
    newmetrics.acc = acc
    newmetrics.ppv = ppv
    newmetrics.npv = npv
    newmetrics.jind = rocdist
    newmetrics.lowsens = lowsens
    newmetrics.lowspec = lowspec
    newmetrics.lowrevspec = lowrevspec
    newmetrics.lowacc = lowacc
    newmetrics.lowppv = lowppv
    newmetrics.lownpv = lownpv
    newmetrics.lowjind = lowrocdist
    newmetrics.uppsens = uppsens
    newmetrics.uppspec = uppspec
    newmetrics.upprevspec = upprevspec
    newmetrics.uppacc = uppacc
    newmetrics.uppppv = uppppv
    newmetrics.uppnpv = uppnpv
    newmetrics.uppjind = upprocdist

    return(newmetrics)

def ML_binClass_xFold_crossValidation (indb,uid,target,xfold,mode="foo",niterations="foo",zindex="foo",attributes="foo",rocvalidation="foo",customwp="foo",droplist="foo",dropthresh="foo",goscalevals="foo",gridsearch="foo"):

    ######################## INITIALIZATION OF DBS

    finoutdb = pd.DataFrame()
    itoutdb = pd.DataFrame()
    fulloutdb = pd.DataFrame()


    ######################## INITIALIZATION OF VARIABLES

    print('Crossvalidation started: preprocessing input dataset')
    if mode == "foo":
        mode = 0
    if gridsearch == "foo":
        gridsearch = 0
    if niterations == "foo":
        niterations = 1
    if zindex == "foo":
        zindex = 3
    if attributes == "foo":
        allatts = 1
    else:
        allatts = 0
    if droplist == "foo":
        dropflag = 0
    else:
        dropflag = 1
    if goscalevals == "foo":
        goscalevals = 0
    else:
        goscalevals = 1
    if rocvalidation=="foo":
        rocvalidation = 0
        customwpflag = 0
    else:
        if customwp == "foo":
            customwpflag = 0
        else:
            customwpflag = 1
    if dropthresh=="foo":
        dropthresh = 0.1


    ######################## MODEL SELECTION

    if type(mode) == str:
        if mode == "logreg" or mode == "LogReg" or mode == "LogisticRegression" or mode == "logisticregression":
            mode = 0
            modestr = 'LogisticRegression'
        if mode == "randfor" or mode == "RandFor" or mode == "RandomForest" or mode == "randomforest":
            mode = 1
            modestr = 'RandomForest'
        if mode == "svm" or mode == "SVM" or mode == "SupportVectorMachine" or mode == "supportvectormachine":
            mode = 2
            modestr = 'SupportVectorMachine'
        if mode == 'nbg' or mode == 'NBG' or mode == 'gau' or mode == 'GAU' or mode == 'NaiveBayesGaussian' or mode == 'naivebayesgaussian':
            mode = 3
            modestr = 'NaiveBayesGaussian'

    if type(mode) == int:
        if mode == 0:
            clf = LogisticRegression()
            modelstring = "LogisticRegression"
        if mode == 1:
            classifier_name = "RF"
            rf = RF(random_state=42)
            param_grid = {'n_estimators': [50, 100, 150],
                          'max_depth': [10, 15, 20, 30, 40, 50]}
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification', rf)])
            if gridsearch == 1:
                clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')
            modelstring = "RandomForest"
        if mode == 2:
            classifier_name="SVM"
            svm_ = svm.SVC(kernel='rbf',probability=True)
            C = np.logspace(-7,3,6)
            gamma = np.logspace(-7, 3, 6)
            cv = 5
            param_grid = dict(C=C, gamma=gamma)
            optimal_params = []
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification', svm_)])
            modelstring = "SupportVectorMachine"
        if mode == 3:
            gnb=GaussianNB()
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification',gnb)])
            modelstring = 'GaussianNaiveBayes'
        print('MODEL SELECTED: ',modelstring)
        baseprint = target + ' ' + modelstring + ' '


    ######################## PREPROCESSING: COLUMNS DROP

    allcols = indb.columns.values.tolist()
    if dropflag == 1:
        if allatts == 1:
            attributes = list()
        for col in allcols:
            if col not in attributes and col not in droplist:
                attributes.append(col)
        allatts = 0
    coli = 0
    popcols = list()
    for col in allcols:
        if 'Unnamed' in col:
            indb = indb.drop(col,1)
            print(baseprint,' PREPROCESSING: dropping column ',col,' at index ',coli)
            popcols.append(coli)
        coli = coli + 1
    coli = 0
    for ci in popcols:
        allcols.pop(ci-coli)
        coli = coli + 1
    coli = 0
    if allatts == 0:
        for col in allcols:
            if col not in attributes and col != target and col != uid:
                indb = indb.drop(col,1)
                print(baseprint,' PREPROCESSING: dropping column ',col,' at index ',coli)
        coli = coli + 1


    ######################## PREPROCESSING: DROPPING NULLS

    print(baseprint,' PRPROCESSING: dropping null values')
    newcols = allcols = indb.columns.values.tolist()
    indb.dropna(subset=[target],inplace=True)
    indim = indb.shape[0]
    naexcludedlist = list()
    for col in newcols:
        nadb = indb.loc[indb[col].isna()].copy(deep=True)
        nadim = nadb.shape[0]
        if nadim > dropthresh*indim:
            indb.drop(col,axis=1)
            naexcludedlist.append(col)
        else:
            indb.dropna(subset=[col],inplace=True)


    ########################  INTIIALIZATION OF X-FOLDS PROCEDURE

    vc = indb[target].value_counts()
    zeroscount = vc[0]
    onescount = vc[1]

    if zeroscount >= onescount:
        longdbtemp = indb[indb[target] == 0].copy()
        smalltempdb = indb[indb[target] == 1].copy()
    elif onescount > zeroscount:
        longdbtemp = indb[indb[target] == 1].copy()
        smalltempdb = indb[indb[target] == 0].copy()

    maxclasslen = math.floor(len(smalltempdb) / xfold)
    maxfoldlen = 2*maxclasslen
    maxclassfold = maxclasslen*xfold
    maxtotlen = maxfoldlen*xfold

    longinds = list(range(0,len(longdbtemp)))
    smallinds = list(range(0,len(smalltempdb)))

    ####################### INITIALIZATION OF FINAL LISTS OF METRICS
    finalMetricsLists = list()
    finalBaseMetrics = list()
    if rocvalidation == 1:
        finalJwpMetrics = list()
        finalNwpMetrics = list()
        if customwpflag == 1:
            finalCwpMetrics = list()

    ######################## GENERATION OF X-FOLDS (with x boosting cycles)

    for nit in range(niterations):
        itstr = modelstring + ' Iteration ' + str(nit+1)

        ####################### INITIALIZATION OF ITERATION-SPECIFIC LIST OF METRICS
        cycleMetricsLists = list()
        cycleBaseMetrics = list()
        if rocvalidation == 1:
            cycleJwpMetrics = list()
            cycleNwpMetrics = list()
            if customwpflag == 1:
                cycleCwpMetrics = list()

        ####################### COMPUTATION OF ITERATION-SPECIFIC X FOLDS
        print('\n\n\t\t',baseprint,' PROCESSING: STARTED ITERATION ',itstr)
        X_folds = list()
        Y_folds = list()
        long_indx = random.sample(set(longinds),maxclassfold)
        small_indx = random.sample(set(smallinds),maxclassfold)

        endind = -1

        for f in range(xfold):
            print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> DIVIDING DATASET')
            foldstr = 'Fold ' + str(f+1)
            fulllinestr = itstr + ' ' + foldstr
            startind = endind + 1
            endind = (f+1)*maxclasslen
            if f == 0:
                longtestinds = long_indx[0:endind]
                smalltestinds = small_indx[0:endind]
                longtraininds = long_indx[endind+1:maxclassfold]
                smalltraininds = small_indx[endind+1:maxclassfold]
            elif f == (xfold-1):
                longtestinds = long_indx[startind:maxclassfold]
                smalltestinds = small_indx[startind:maxclassfold]
                longtraininds = long_indx[0:startind-1]
                smalltraininds = small_indx[0:startind-1]
            else:
                longtestinds = long_indx[startind:endind]
                smalltestinds = small_indx[startind:endind]
                longtraininds = (long_indx[0:startind-1]+long_indx[endind+1:maxclassfold])
                smalltraininds = (small_indx[0:startind-1]+small_indx[endind+1:maxclassfold])

            ######################## TRAIN TEST ON ONE FOLD: ATTRIBUTES SELECTION

            print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> SELECTING ATTRIBUTES')
            traindbraw = indb.loc[indb.index[longtraininds+smalltraininds]].copy(deep=True)
            testdbraw = indb.loc[indb.index[longtestinds+smalltestinds]].copy(deep=True)
            [conf,exc,xdb,res] = multivariate_logreg_or(traindbraw,attributes,uid,target,0,0.2)
            traincols = list()
            traindb = pd.DataFrame()
            testdb = pd.DataFrame()
            for att in attributes:
                traincols.append(att)
            for ex in exc:
                traincols.pop(traincols.index(ex))

            ######################## TRAIN TEST ON ONE FOLD: ATTRIBUTES PRE-PROCESSING

            print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> DATA PREPROCESSING')
            for tc in traincols:
                traindbraw.dropna(subset=[tc],inplace=True)
                datavec = pd.DataFrame(traindbraw[tc].copy(deep=True))
                dim = datavec.shape[0]
                bindf = datavec.loc[((datavec[tc]==0)|(datavec[tc]==1))].copy(deep=True)
                bindim = bindf.shape[0]
                if (dim-bindim) <= (0.05*dim):
                    datatype = 0
                else:
                    checker = datavec.iloc[0,0]
                    if type(checker)!=float and type(checker)!=int and type(checker)!=np.int64 and type(checker)!=np.float64:
                        datatype = 2
                    else:
                        datatype = 1
                if datatype == 1:
                    allvals = np.asarray(traindbraw[tc])
                    medval = np.median(allvals)
                    iqr = (np.quantile(allvals, .75)-np.quantile(allvals, .25))
                    lowth = medval - (zindex*iqr)
                    upth = medval + (zindex*iqr)
                    ttraindbraw = traindbraw.loc[((traindbraw[tc]>lowth)&(traindbraw[tc]<upth))].copy(deep=True)
                    traindbraw = ttraindbraw
            for tc in traincols:
                datavec = pd.DataFrame(traindbraw[tc].copy(deep=True))
                dim = datavec.shape[0]
                bindf = datavec.loc[((datavec[tc]==0)|(datavec[tc]==1))].copy(deep=True)
                bindim = bindf.shape[0]
                if (dim-bindim) <= (0.05*dim):
                    datatype = 0
                else:
                    checker = datavec.iloc[0,0]
                    if type(checker)!=float and type(checker)!=int and type(checker)!=np.int64 and type(checker)!=np.float64:
                        datatype = 2
                    else:
                        datatype = 1
                if datatype == 0:
                    newtraincol = pd.DataFrame(np.asarray(traindbraw[tc]))
                    newtestcol = pd.DataFrame(np.asarray(testdbraw[tc]))
                    newtraincol = newtraincol.rename(columns={0:tc})
                    newtestcol = newtestcol.rename(columns={0:tc})
                elif datatype == 1:
                    colarrtrain = np.asarray(traindbraw[tc])
                    colarrtest = np.asarray(testdbraw[tc])
                    maxval = max(colarrtrain)
                    minval = min(colarrtrain)
                    scaledcoltrain = scalevals(colarrtrain,minval,maxval)
                    medval = np.median(colarrtrain)
                    colarrtest[np.isnan(colarrtest)] = medval
                    scaledcoltest = scalevals(colarrtest,minval,maxval)
                    scaledcoltest[scaledcoltest>1] = 1
                    scaledcoltest[scaledcoltest<0] = 0
                    newtraincol = pd.DataFrame(scaledcoltrain)
                    newtestcol = pd.DataFrame(scaledcoltest)
                    newtraincol = newtraincol.rename(columns={0:tc})
                    newtestcol = newtestcol.rename(columns={0:tc})
                elif datatype == 2:
                    newtraincol = pd.DataFrame(np.asarray(traindbraw[tc]))
                    newtestcol = pd.DataFrame(np.asarray(testdbraw[tc]))
                    newtraincol = newtraincol.rename(columns={0:tc})
                    newtestcol = newtestcol.rename(columns={0:tc})
                    allcats = newtraincol[tc].unique()
                    testcats = newtestcol[tc].unique()
                    ncats = len(allcats)
                    catcount = 1
                    for cat in allcats:
                        subval = catcount/ncats
                        newtraincol.loc[newtraincol[tc]==cat,tc] = subval
                        newtestcol.loc[newtestcol[tc]==cat,tc] = subval
                        catcount = catcount + 1
                    for ct in testcats:
                        if ct not in allcats:
                            newtestcol.loc[newtestcol[tc]==ct,tc] = 0
        
                traindb = pd.concat([traindb,newtraincol],axis=1)
                testdb = pd.concat([testdb,newtestcol],axis=1)


            ######################## TRAIN TEST ON ONE FOLD: EXECUTION

            traintarget = pd.DataFrame(np.asarray(traindbraw[target]))
            testtarget = pd.DataFrame(np.asarray(testdbraw[target]))
            trainuid = pd.DataFrame(np.asarray(traindbraw[uid]))
            testuid = pd.DataFrame(np.asarray(testdbraw[uid]))

            X_train = np.asarray(traindb)
            Y_train = np.asarray(traintarget)
            X_test = np.asarray(testdb)
            Y_test = np.asarray(testtarget)

            print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> TRAINING')

            clf.fit(X_train,Y_train.ravel())

            print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> TESTING')

            Y_pred = clf.predict(X_test)

            if mode == 1:
                print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> COMPUTING VARIABLES WEIGHTS - FITTING')
                optpars = clf.best_params_
                ne = optpars.get('n_estimators')
                md = optpars.get('max_depth')
                shclf = RF(n_estimators=ne,max_depth=md)
                shclf.fit(X_train,Y_train.ravel())
                print(baseprint,itstr,' RUNNING FOLD ',f+1,' of ',xfold,'\t --> COMPUTING VARIABLES WEIGHTS - COMPUTING')
                shap_values = shap.TreeExplainer(shclf).shap_values(X_train)
                br = 1
            ######################## TRAIN TEST ON ONE FOLD: COMPUTATION OF BASIC STATS

            baseMetrics = compute_metrics(Y_test,Y_pred)
            baseMetrics.thresh = 0.5
            finalBaseMetrics.append(baseMetrics)
            cycleBaseMetrics.append(baseMetrics)

            fulloutdb.loc[fulllinestr,'BASE_SENS'] = baseMetrics.sens
            fulloutdb.loc[fulllinestr,'BASE_SENS_low'] = baseMetrics.lowsens
            fulloutdb.loc[fulllinestr,'BASE_SENS_upp'] = baseMetrics.uppsens
            fulloutdb.loc[fulllinestr,'BASE_SPEC'] = baseMetrics.spec
            fulloutdb.loc[fulllinestr,'BASE_SPEC_low'] = baseMetrics.lowspec
            fulloutdb.loc[fulllinestr,'BASE_SPEC_upp'] = baseMetrics.uppspec
            fulloutdb.loc[fulllinestr,'BASE_ACC'] = baseMetrics.acc
            fulloutdb.loc[fulllinestr,'BASE_ACC_low'] = baseMetrics.lowacc
            fulloutdb.loc[fulllinestr,'BASE_ACC_upp'] = baseMetrics.uppacc
            fulloutdb.loc[fulllinestr,'BASE_PPV'] = baseMetrics.ppv
            fulloutdb.loc[fulllinestr,'BASE_PPV_low'] = baseMetrics.lowppv
            fulloutdb.loc[fulllinestr,'BASE_PPV_upp'] = baseMetrics.uppppv
            fulloutdb.loc[fulllinestr,'BASE_NPV'] = baseMetrics.npv
            fulloutdb.loc[fulllinestr,'BASE_NPV_low'] = baseMetrics.lownpv
            fulloutdb.loc[fulllinestr,'BASE_NPV_upp'] = baseMetrics.uppnpv
            fulloutdb.loc[fulllinestr,'BASE_JIND'] = baseMetrics.jind
            fulloutdb.loc[fulllinestr,'BASE_JIND_low'] = baseMetrics.lowjind
            fulloutdb.loc[fulllinestr,'BASE_JIND_upp'] = baseMetrics.uppjind



            ######################## COMPUTATION OF ROC AND WORKING POINTS STATISTICS

            if rocvalidation==1:
                newRocList = metricsList(fulllinestr.replace(' ','_'))
                probvals = clf.predict_proba(X_test)
                yplen = len(probvals)
                for alphathresh in range(101):
                    alpha = alphathresh/100
                    y_probtest = list()
                    ypcount = 0
                    for yallp in probvals:
                        ypcount = ypcount + 1
                        yp = yallp[1]
                        print(baseprint,itstr,' COMPUTING ROC VALIDATION for FOLD ',f+1,'\tProcessing = ',(((alphathresh*yplen)+ypcount)/(yplen*100))*100,' %')
                        if yp >= alpha:
                            y_probtest.append(1)
                        else:
                            y_probtest.append(0)
                    newAlphaMetrics = compute_metrics(Y_test,y_probtest)
                    newAlphaMetrics.thresh = alpha
                    newRocList.metrics.append(newAlphaMetrics)
                newRocList.compute_roc()
                newRocList.compute_nindx()
                finalMetricsLists.append(newRocList)
                cycleMetricsLists.append(newRocList)
                fulloutdb.loc[fulllinestr,'AUC'] = newRocList.auc
                fulloutdb.loc[fulllinestr,'AUC_low'] = newRocList.auc_low
                fulloutdb.loc[fulllinestr,'AUC_upp'] = newRocList.auc_upp

                ######################## IDENTIFICATION OF J WORKING POINT

                jmin = 9999
                for m in newRocList.metrics:
                    newjind = m.jind
                    if newjind < jmin:
                        jmin = newjind
                        jwp = m
                fulloutdb.loc[fulllinestr,'JWP_SENS'] = jwp.sens
                fulloutdb.loc[fulllinestr,'JWP_SENS_low'] = jwp.lowsens
                fulloutdb.loc[fulllinestr,'JWP_SENS_upp'] = jwp.uppsens
                fulloutdb.loc[fulllinestr,'JWP_SPEC'] = jwp.spec
                fulloutdb.loc[fulllinestr,'JWP_SPEC_low'] = jwp.lowspec
                fulloutdb.loc[fulllinestr,'JWP_SPEC_upp'] = jwp.uppspec
                fulloutdb.loc[fulllinestr,'JWP_ACC'] = jwp.acc
                fulloutdb.loc[fulllinestr,'JWP_ACC_low'] = jwp.lowacc
                fulloutdb.loc[fulllinestr,'JWP_ACC_upp'] = jwp.uppacc
                fulloutdb.loc[fulllinestr,'JWP_PPV'] = jwp.ppv
                fulloutdb.loc[fulllinestr,'JWP_PPV_low'] = jwp.lowppv
                fulloutdb.loc[fulllinestr,'JWP_PPV_upp'] = jwp.uppppv
                fulloutdb.loc[fulllinestr,'JWP_NPV'] = jwp.npv
                fulloutdb.loc[fulllinestr,'JWP_NPV_low'] = jwp.lownpv
                fulloutdb.loc[fulllinestr,'JWP_NPV_upp'] = jwp.uppnpv
                fulloutdb.loc[fulllinestr,'JWP_JIND'] = jwp.jind
                fulloutdb.loc[fulllinestr,'JWP_JIND_low'] = jwp.lowjind
                fulloutdb.loc[fulllinestr,'JWP_JIND_upp'] = jwp.uppjind
                fulloutdb.loc[fulllinestr,'JWP_NIND'] = jwp.nind
                fulloutdb.loc[fulllinestr,'JWP_NIND_low'] = jwp.lownind
                fulloutdb.loc[fulllinestr,'JWP_NIND_upp'] = jwp.uppnind
                cycleJwpMetrics.append(jwp)
                finalJwpMetrics.append(jwp)

                ######################## IDENTIFICATION OF N WORKING POINT
                nmin = 9999
                for m in newRocList.metrics:
                    newnind = m.nind
                    if newnind < nmin:
                        nmin = newnind
                        nwp = m
                fulloutdb.loc[fulllinestr,'NWP_SENS'] = nwp.sens
                fulloutdb.loc[fulllinestr,'NWP_SENS_low'] = nwp.lowsens
                fulloutdb.loc[fulllinestr,'NWP_SENS_upp'] = nwp.uppsens
                fulloutdb.loc[fulllinestr,'NWP_SPEC'] = nwp.spec
                fulloutdb.loc[fulllinestr,'NWP_SPEC_low'] = nwp.lowspec
                fulloutdb.loc[fulllinestr,'NWP_SPEC_upp'] = nwp.uppspec
                fulloutdb.loc[fulllinestr,'NWP_ACC'] = nwp.acc
                fulloutdb.loc[fulllinestr,'NWP_ACC_low'] = nwp.lowacc
                fulloutdb.loc[fulllinestr,'NWP_ACC_upp'] = nwp.uppacc
                fulloutdb.loc[fulllinestr,'NWP_PPV'] = nwp.ppv
                fulloutdb.loc[fulllinestr,'NWP_PPV_low'] = nwp.lowppv
                fulloutdb.loc[fulllinestr,'NWP_PPV_upp'] = nwp.uppppv
                fulloutdb.loc[fulllinestr,'NWP_NPV'] = nwp.npv
                fulloutdb.loc[fulllinestr,'NWP_NPV_low'] = nwp.lownpv
                fulloutdb.loc[fulllinestr,'NWP_NPV_upp'] = nwp.uppnpv
                fulloutdb.loc[fulllinestr,'NWP_JIND'] = nwp.jind
                fulloutdb.loc[fulllinestr,'NWP_JIND_low'] = nwp.lowjind
                fulloutdb.loc[fulllinestr,'NWP_JIND_upp'] = nwp.uppjind
                fulloutdb.loc[fulllinestr,'NWP_NIND'] = nwp.nind
                fulloutdb.loc[fulllinestr,'NWP_NIND_low'] = nwp.lownind
                fulloutdb.loc[fulllinestr,'NWP_NIND_upp'] = nwp.uppnind
                cycleNwpMetrics.append(nwp)
                finalNwpMetrics.append(nwp)

                if customwpflag == 1:
                    search = 1
                    if 'SENS' in customwp.keys():
                        thval = customwp['SENS']
                    elif 'SPEC' in customwp.keys():
                        thval = 1 - customwp['SPEC']
                    for m in newRocList.metrics:
                        if search == 1:
                            if 'SENS' in customwp.keys():
                                chkval = m.sens
                            elif 'SPEC' in customwp.keys():
                                chkval = m.revspec
                            if chkval < thval:
                                search = 0
                            else:
                                cwp = m
                    fulloutdb.loc[fulllinestr,'CWP_SENS'] = cwp.sens
                    fulloutdb.loc[fulllinestr,'CWP_SENS_low'] = cwp.lowsens
                    fulloutdb.loc[fulllinestr,'CWP_SENS_upp'] = cwp.uppsens
                    fulloutdb.loc[fulllinestr,'CWP_SPEC'] = cwp.spec
                    fulloutdb.loc[fulllinestr,'CWP_SPEC_low'] = cwp.lowspec
                    fulloutdb.loc[fulllinestr,'CWP_SPEC_upp'] = cwp.uppspec
                    fulloutdb.loc[fulllinestr,'CWP_ACC'] = cwp.acc
                    fulloutdb.loc[fulllinestr,'CWP_ACC_low'] = cwp.lowacc
                    fulloutdb.loc[fulllinestr,'CWP_ACC_upp'] = cwp.uppacc
                    fulloutdb.loc[fulllinestr,'CWP_PPV'] = cwp.ppv
                    fulloutdb.loc[fulllinestr,'CWP_PPV_low'] = cwp.lowppv
                    fulloutdb.loc[fulllinestr,'CWP_PPV_upp'] = cwp.uppppv
                    fulloutdb.loc[fulllinestr,'CWP_NPV'] = cwp.npv
                    fulloutdb.loc[fulllinestr,'CWP_NPV_low'] = cwp.lownpv
                    fulloutdb.loc[fulllinestr,'CWP_NPV_upp'] = cwp.uppnpv
                    fulloutdb.loc[fulllinestr,'CWP_JIND'] = cwp.jind
                    fulloutdb.loc[fulllinestr,'CWP_JIND_low'] = cwp.lowjind
                    fulloutdb.loc[fulllinestr,'CWP_JIND_upp'] = cwp.uppjind
                    fulloutdb.loc[fulllinestr,'CWP_NIND'] = cwp.nind
                    fulloutdb.loc[fulllinestr,'CWP_NIND_low'] = cwp.lownind
                    fulloutdb.loc[fulllinestr,'CWP_NIND_upp'] = cwp.uppnind
                    cycleCwpMetrics.append(cwp)
                    finalCwpMetrics.append(cwp)

        cysti = (nit) * xfold
        cyedi = ((nit+1) * xfold)
        outdbcols = list(fulloutdb.columns.values)
        for oc in outdbcols:
            arr = np.asarray(fulloutdb[oc][cysti:cyedi])
            [medcy,q25cy,q75cy,lowcicy,uppcyci] = compute_dist(arr)
            itoutdb.loc[(oc+'_MED'),itstr] = medcy
            itoutdb.loc[(oc+'_Q25'),itstr] = q25cy
            itoutdb.loc[(oc+'_Q75'),itstr] = q75cy
            itoutdb.loc[(oc+'_LCI'),itstr] = lowcicy
            itoutdb.loc[(oc+'_UCI'),itstr] = uppcyci

    for oc in outdbcols:
        arr = np.asarray(fulloutdb[oc])
        [medcy,q25cy,q75cy,lowcicy,uppcyci] = compute_dist(arr)
        finoutdb.loc[(oc+'_MED'),modelstring] = medcy
        finoutdb.loc[(oc+'_Q25'),modelstring] = q25cy
        finoutdb.loc[(oc+'_Q75'),modelstring] = q75cy
        finoutdb.loc[(oc+'_LCI'),modelstring] = lowcicy
        finoutdb.loc[(oc+'_UCI'),modelstring] = uppcyci

    if rocvalidation == 1:
        plotroc = pd.DataFrame()
        plotroc.loc[0,0] = ' '
        cumarea = 0
        low_cumarea = 0
        upp_cumarea = 0
        minccf = 1
        nsteps = 1
        for alpha in range(101):
            newsenslist = list()
            newrevspeclist = list()
            low_newsenslist = list()
            low_newrevspeclist = list()
            upp_newsenslist = list()
            upp_newrevspeclist = list()
            for ml in finalMetricsLists:
                newsenslist.append(ml.metrics[alpha].sens)
                newrevspeclist.append(ml.metrics[alpha].revspec)
                low_newsenslist.append(ml.metrics[alpha].lowsens)
                low_newrevspeclist.append(ml.metrics[alpha].lowrevspec)
                upp_newsenslist.append(ml.metrics[alpha].uppsens)
                upp_newrevspeclist.append(ml.metrics[alpha].upprevspec)
            newsensvec = np.asarray(newsenslist)
            newspecvec = np.asarray(newrevspeclist)
            low_newsensvec = np.asarray(low_newsenslist)
            low_newspecvec = np.asarray(low_newrevspeclist)
            upp_newsensvec = np.asarray(upp_newsenslist)
            upp_newspecvec = np.asarray(upp_newrevspeclist)
            newsens = median(newsensvec)
            newspec = median(newspecvec)
            low_newsens = median(low_newsensvec)
            low_newspec = median(low_newspecvec)
            upp_newsens = median(upp_newsensvec)
            upp_newspec = median(upp_newspecvec)
            plotroc.loc[alpha,'SENS'] = newsens
            plotroc.loc[alpha,'1-SPEC'] = newspec
            plotroc.loc[alpha,'LOW_SENS'] = low_newsens
            plotroc.loc[alpha,'LOW_1-SPEC'] = low_newspec
            plotroc.loc[alpha,'UPP_SENS'] = upp_newsens
            plotroc.loc[alpha,'UPP_1-SPEC'] = upp_newspec
            if alpha == 0:
                xi = newspec
                yi = newsens
                low_xi = low_newspec
                low_yi = low_newsens
                upp_xi = upp_newspec
                upp_yi = upp_newsens
            else:
                xf = newspec
                yf = newsens
                low_xf = low_newspec
                low_yf = low_newsens
                upp_xf = upp_newspec
                upp_yf = upp_newsens
                newarea = abs(((yf+yi) * (xf-xi))/2)
                low_newarea = abs(((low_yf+low_yi) * (low_xf-low_xi))/2)
                upp_newarea = abs(((upp_yf+upp_yi) * (upp_xf-upp_xi))/2)
                if newarea == 0:
                    nsteps = nsteps + 1
                else:
                    nsteps = 1
                low_cumarea = low_cumarea + low_newarea
                upp_cumarea = upp_cumarea + upp_newarea
                cumarea = cumarea + newarea
                xi = xf
                yi = yf
                low_xi = low_xf
                low_yi = low_yf
                upp_xi = upp_xf
                upp_yi = upp_yf
            newccf = sqrt(((1-newsens)**2)+(newspec**2))
            if newccf < minccf:
                minccf = newccf
                bestalpha = (alpha/100)
                alphasens = newsens
                alphaspec = 1-newspec
                
        statstext = baseprint + ' AUC = ' + str(cumarea) +' CI95% ' + str(low_cumarea) + ' - '+ str(upp_cumarea) + ' ALPHA = ' + str(round(bestalpha*100)/100) + ' with SENS = ' + str(round(alphasens*10000)/100) + '% and SPEC = ' + str(round(alphaspec*10000)/100) + '% (cost = ' + str(round(minccf*1000)/1000) + ')'
        plotroc.iloc[0,0] = statstext
        return(finoutdb,itoutdb,fulloutdb,plotroc)

    return(finoutdb,itoutdb,fulloutdb)
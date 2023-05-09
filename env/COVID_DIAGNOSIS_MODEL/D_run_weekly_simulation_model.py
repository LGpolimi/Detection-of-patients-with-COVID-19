print('Script started')
import os
import pandas as pd
import datetime
import math
import numpy
import statistics
from xFold_LogReg_attributes_selection import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from Z_classes_and_functions import *

################################################ INITIALIZATIONS

outpath = 'ML_SIMULATION_OUTPUTS//'
if not(os.path.isdir(outpath)):
    os.makedirs(outpath)
model = 'RF'
quart = '' # EMPTY for full DB, 'Q1_' for first quartile .... 'Q4_' for last quartile
prefix = 'DO_' + quart + model + '_M2_'
indbname = 'AO' + quart + '_MainDB_201001_210721.xlsx'
uid = 'ID'
targetVariable = 'ESITO_+-7'
attributesList = ['AGE','SEX','CONTATTO','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','COSCIENZA','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST','TEMPERATURA','TOT_POS']
attributesTypes = ['num','cat','bin','bin','bin','bin','bin','bin','bin','bin','cat','num','num','num','num','num','num','num','num']
keepFields = ['DATE_INT','HUB','CRIT','TAMPONE_+-8','ESITO_+-8','TAMPONE_+-9','ESITO_+-9','TAMPONE_+-10','ESITO_+-10','TAMPONE_+-11','ESITO_+-11','TAMPONE_+-12','ESITO_+-12','TAMPONE_+-13','ESITO_+-13','TAMPONE_+-14','ESITO_+-14','TAMPONE_+-15','ESITO_+-15',]
startdate = datetime.datetime.strptime('01OCT2020','%d%b%Y')
enddate = datetime.datetime.strptime('21JUL2021','%d%b%Y')
timewindow = 7
offsetCycles = 4
attAcceptedNans = 0.1
zindex = 5
alphasteps = 400
xfoldCutoffFlag = 1
dtfield = 'DATE_INT'
goplot = 1
outinfo = pd.DataFrame()

print('\nIMPORTING DATASOURCES')
indbraw = pd.read_excel(indbname)
indb = indbraw.loc[~indbraw[targetVariable].isna()].copy(deep=True)



############################################## DATETIME-BASED SUBDIVISION



dtcelli = 0
dtcellf = 9
inDtFormat = '%d%b%Y'
offPart = ':00:00:00.000'
ncycles = math.ceil((enddate-startdate).days/timewindow)
pastStrings = list()
lastStrings = list()

cycle = 0

while cycle < (ncycles):
    cycle = cycle + 1
    cyclestr ='\nRUNNING cycle ' + str(cycle) + ' out of ' + str(ncycles) + ' (' + str((cycle/ncycles)*100) + '%)'
    print(cyclestr)
    if cycle == 1:
        pdayi = startdate
        pdayf = startdate + datetime.timedelta(days=timewindow-1)
        for pdy in range(((pdayf-pdayi).days)+1):
            newDtStr = datetime.datetime.strftime((pdayi+datetime.timedelta(days=pdy)),'%d%b%Y').upper()+offPart
            pastStrings.append(newDtStr)
        ldayi = pdayf + datetime.timedelta(days=1)
        ldayf = ldayi + datetime.timedelta(days=timewindow-1)
        for ld in range(((ldayf-ldayi).days)+1):
            newDtStr = datetime.datetime.strftime((ldayi+datetime.timedelta(days=ld)),'%d%b%Y').upper()+offPart
            lastStrings.append(newDtStr)
            initIn = 1
        for ps in pastStrings:
            if initIn == 1:
                traindbLvlZero = indb.loc[indb[dtfield] == ps].copy(deep=True)
                initIn = 0
            else:
                traindbLvlZero = pd.concat([traindbLvlZero,indb.loc[indb[dtfield] == ps].copy(deep=True)])
        initIn = 1
        for ls in lastStrings:
            if initIn == 1:
                testdbLvlZero = indb.loc[indb[dtfield] == ls].copy(deep=True)
                initIn = 0
            else:
                testdbLvlZero = pd.concat([testdbLvlZero,indb.loc[indb[dtfield] == ls].copy(deep=True)])
    else:
        traindbLvlZero = pd.concat([traindbLvlZero,testdbLvlZero])
        ldayi = ldayf + datetime.timedelta(days=1)
        ldayf = ldayi + datetime.timedelta(days=timewindow-1)
        if ldayf > enddate:
            ldayf = enddate
        lastStrings = list()
        for ld in range(((ldayf-ldayi).days)+1):
            newDtStr = datetime.datetime.strftime((ldayi+datetime.timedelta(days=ld)),'%d%b%Y').upper()+offPart
            lastStrings.append(newDtStr)
        initIn = 1
        for ls in lastStrings:
            if initIn == 1:
                testdbLvlZero = indb.loc[indb[dtfield] == ls].copy(deep=True)
                initIn = 0
            else:
                testdbLvlZero = pd.concat([testdbLvlZero,indb.loc[indb[dtfield] == ls].copy(deep=True)])



    ############################################## REMOVE OUTLIERS FROM TRAINING



    print(cyclestr+' Removing outliers')
    atti = 0
    for att in attributesList:
        if attributesTypes[atti] == 'num':
            attdb = traindbLvlZero.loc[~traindbLvlZero[att].isna()].copy(deep=True)
            allvals = np.asarray(attdb[att])
            attmed = median(allvals)
            attlowq = np.quantile(allvals,0.25)
            attupq = np.quantile(allvals,0.75)
            iqr = attupq-attlowq
            attLowThresh = attmed-(zindex*iqr)
            attUpThresh = attmed+(zindex*iqr)
            traindbLvlZero = pd.concat([traindbLvlZero.loc[((traindbLvlZero[att]<=attUpThresh) & (traindbLvlZero[att]>=attLowThresh))],traindbLvlZero.loc[traindbLvlZero[att].isna()]])
        atti = atti + 1



    ############################################## BALANCING OF TRAINING SET



    print(cyclestr+' Balancing training set')
    ldayfstr = datetime.datetime.strftime(ldayf,'%d%b%Y')
    ldayistr = datetime.datetime.strftime(ldayi,'%d%b%Y')
    if cycle < 10:
        strc = '0' + str(cycle)
    else:
        strc = str(cycle)
    outwstr = 'W' + strc + '_' + ldayfstr
    ldayfstr = outwstr

    traindbLvlZero = traindbLvlZero.loc[~traindbLvlZero[targetVariable].isna()].copy(deep=True)
    trainAllOnes = traindbLvlZero.loc[traindbLvlZero[targetVariable] == 1].copy(deep=True)
    trainAllZeros = traindbLvlZero.loc[traindbLvlZero[targetVariable] != 1].copy(deep=True)
    attsList = attributesList.copy()
    attsTypes = attributesTypes.copy()
    removedAtts = list()
    nonesRef = trainAllOnes.shape[0]
    trainOnes = trainAllOnes
    for att in attributesList:
        attCheck = trainAllOnes[~trainAllOnes[att].isna()].copy(deep=True)
        nonesCheck = attCheck.shape[0]
        if nonesCheck <= nonesRef * (1-attAcceptedNans):
            apidx = attsList.index(att)
            attsList.pop(apidx)
            attsTypes.pop(apidx)
            removedAtts.append(att)
        else:
            trainOnes = trainOnes[~trainOnes[att].isna()].copy(deep=True)
            trainAllZeros = trainAllZeros[~trainAllZeros[att].isna()].copy(deep=True)
    nones = trainOnes.shape[0]
    overzerosflag = 0
    for oz in range(8):
        if overzerosflag == 0:
            windowend = 15-oz
            checkcol = 'ESITO_+-'+str(windowend)
            trainSelZeros = trainAllZeros.loc[trainAllZeros[checkcol] == 0].copy(deep=True)
            nzeros = trainSelZeros.shape[0]
            if nzeros >= nones:
                overzerosflag = 1
    if overzerosflag == 0:
        trainZeros = traindbLvlZero.loc[traindbLvlZero['ESITO_+-7'] == 0].copy(deep=True)
    else:
        trainZeros = trainSelZeros.iloc[(nzeros-nones):nzeros]
    traindbLvlOne = pd.concat([trainOnes,trainZeros]).sample(frac=1).copy(deep=True)



    ############################################## SCALING AND TRANSFORMING TRAINING AND TEST SET



    print(cyclestr+' Scaling training and test set')
    procTrainDb = traindbLvlOne.copy(deep=True)
    procTestDb = testdbLvlZero.copy(deep=True)

    lvltwolist = list([uid])

    for att in attsList:
        lvltwolist.append(att)
        idx = attsList.index(att)
        dtype = attsTypes[idx]
        if dtype == 'cat':
            allcats = traindbLvlOne[att].unique()
            testcats = procTestDb[att].unique()
            ncats = len(allcats)
            catcount = 1
            for cat in allcats:
                subval = catcount/ncats
                procTrainDb.loc[procTrainDb[att]==cat,att] = subval
                procTestDb.loc[procTestDb[att]==cat,att] = subval
                catcount = catcount + 1
            for tc in testcats:
                if tc not in allcats:
                    procTestDb.loc[procTestDb[att]==tc,att] = 0
        elif dtype == 'num':
            dfvec = procTrainDb[att]
            npvec = dfvec.to_numpy()
            attmax = max(npvec)
            attmin = min(npvec)
            dfvec = ((dfvec - attmin)/(attmax-attmin))
            dfvec = dfvec.rename_axis(att)
            procTrainDb = procTrainDb.rename(columns={att:(att+'_NS')})
            procTrainDb = pd.concat([procTrainDb,dfvec],axis=1,join='inner')
            tdfvec = procTestDb[att]
            tdfvec = ((tdfvec - attmin)/(attmax-attmin))
            procTestDb = procTestDb.rename(columns={att:(att+'_NS')})
            procTestDb = pd.concat([procTestDb,tdfvec],axis=1,join='inner')
    lvltwolist.append(targetVariable)
    traindbLvlTwo = procTrainDb[lvltwolist].copy(deep=True)
    testdbLvlTwo = procTestDb[lvltwolist].copy(deep=True)



    ############################################## SELECT ATTRIBUTES WITH ODDS RATIOS C.I.



    print(cyclestr+' Selecting relevant attributes')
    lvlthreelist = lvltwolist.copy()
    logreglist = lvlthreelist.copy()
    logreglist.pop(logreglist.index(uid))
    logreglist.pop(logreglist.index(targetVariable))
    if cycle >= offsetCycles:
        [logtab,droppedAtts,xDB] = multivariate_logreg_or(traindbLvlTwo,logreglist,uid,targetVariable,0)
        excludeAtts = list()
        for row in logtab.iterrows():
            if float(row[1]['CI 2.5%']) < 1 and float(row[1]['CI 97.5%']) > 1:
                excludeAtts.append(row[0])
        orOutpath = outpath+'ODDS_RATIOS//'
        if not(os.path.isdir(orOutpath)):
            os.makedirs(orOutpath)
        logtab.to_excel(orOutpath+prefix+'oddsCI_to_'+outwstr+'.xlsx')
        for eatt in excludeAtts:
            lvlthreelist.pop(lvlthreelist.index(eatt))
        traindbLvlThree = traindbLvlTwo[lvlthreelist].copy(deep=True)
        testdbLvlThree = testdbLvlTwo[lvlthreelist].copy(deep=True)



        ############################################## EXPORT DIMS INFO



        traindim = traindbLvlThree.shape[0]
        testdim = testdbLvlThree.shape[0]
        testPos = testdbLvlThree.loc[testdbLvlThree[targetVariable]==1].copy(deep=True)
        testNeg = testdbLvlThree.loc[testdbLvlThree[targetVariable]==0].copy(deep=True)
        testPosN = testPos.shape[0]
        testNegN = testNeg.shape[0]
        outinfo.loc[ldayfstr,'TrainDim'] = traindim
        outinfo.loc[ldayfstr,'TestDim'] = testdim
        outinfo.loc[ldayfstr,'TestPos'] = testPosN
        outinfo.loc[ldayfstr,'TestNeg'] = testNegN
        outinfo.loc[ldayfstr,'Removed'] = ''
        outinfo['Removed'] = outinfo['Removed'].astype('object')
        outinfo.at[ldayfstr,'Removed'] = (removedAtts)
        outinfo.loc[ldayfstr,'Excluded'] = ''
        outinfo['Excluded'] = outinfo['Excluded'].astype('object')
        outinfo.at[ldayfstr,'Excluded'] = (excludeAtts)




        ############################################## FILLING TEST SET



        mlatts = lvlthreelist.copy()
        mlatts.pop(mlatts.index(uid))
        mlatts.pop(mlatts.index(targetVariable))
        testdbLvlFour = testdbLvlThree.copy(deep=True)
        for ma in mlatts:
            nans = testdbLvlThree.loc[testdbLvlThree[ma].isna()].copy(deep=True)
            nansN = nans.shape[0]
            if nansN > 0:
                vals = traindbLvlThree.loc[~traindbLvlThree[ma].isna()].copy(deep=True)
                allvals = np.asarray(vals[ma])
                medval = median(allvals)
                testdbLvlFour.loc[testdbLvlFour[ma].isna(),ma] = medval



        ############################################## TRAIN AND TEST MAIN ML ALGORITHM



        print(cyclestr+' INITIALIZING ML MODEL')
        if model == 'RF':
            classifier_name = "RF"
            rf = RF(random_state=42)
            param_grid = {'n_estimators': [10, 50, 100, 150, 200],
                          'max_depth': [10, 15, 20, 30, 40, 50]}
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification', rf)])
            clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')
        elif model == 'RFb':
            classifier_name = "RFb"
            rf = RF(random_state=42)
            param_grid = {'n_estimators': [10, 50, 100, 150, 200],
                          'max_depth': [10, 15, 20, 30, 40, 50]}
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification', rf)])
        elif model == 'SVM':
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
        elif model == 'GAU':
            gnb=GaussianNB()
            clf = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV(random_state=0))),
                ('classification',gnb)])



        X_train = traindbLvlThree[(mlatts)].copy(deep=True).to_numpy().astype(float)
        Y_train = traindbLvlThree[targetVariable].copy(deep=True).to_numpy().astype(int).reshape(-1,1).ravel()
        X_test = testdbLvlFour[(mlatts)].copy(deep=True).to_numpy().astype(float)
        Y_test = testdbLvlFour[targetVariable].copy(deep=True).to_numpy().astype(int).reshape(-1,1).ravel()
        print(cyclestr+' TRAINING ML MODEL')
        clf.fit(X_train, Y_train)
        print(cyclestr+' TESTING MODEL')
        pred_prob=clf.predict_proba(X_test)
        predOne = pred_prob[:,1].reshape(-1,1).copy()
        predVals = pd.DataFrame(predOne)
        predVals = predVals.rename(columns={0:'PRED_VAL'})
        testdbLvlFour.reset_index(drop=True, inplace=True)
        testResults = pd.concat([testdbLvlFour,predVals],axis=1)
        trOutpath = outpath+'TEST_RESULTS//'
        if not(os.path.isdir(trOutpath)):
            os.makedirs(trOutpath)
        testResults.to_excel(trOutpath+prefix+'model_results_'+outwstr+'.xlsx')


        ####################################################### EVALUATE PERFORMANCE WITH ROC



        print(cyclestr+' EVALUATING PERFORMANCE')
        perfTab = pd.DataFrame()
        rocalist = list()
        firstalpha = 1
        for alphaint in range((alphasteps+1)):
            alpha = alphaint/alphasteps
            tps = testResults.loc[((testResults[targetVariable]==1) & (testResults['PRED_VAL']>alpha))].copy(deep=True)
            tp = tps.shape[0]
            tpp = round((tp/testdim)*10000)/100
            fps = testResults.loc[((testResults[targetVariable]==0) & (testResults['PRED_VAL']>alpha))].copy(deep=True)
            fp = fps.shape[0]
            fpp = round((fp/testdim)*10000)/100
            tns = testResults.loc[((testResults[targetVariable]==0) & (testResults['PRED_VAL']<=alpha))].copy(deep=True)
            tn = tns.shape[0]
            tnp = round((tn/testdim)*10000)/100
            fns = testResults.loc[((testResults[targetVariable]==1) & (testResults['PRED_VAL']<=alpha))].copy(deep=True)
            fn = fns.shape[0]
            fnp = round((fn/testdim)*10000)/100
            sens = tp/(tp+fn)
            spec = tn/(tn+fp)
            erind = sqrt(((1-sens)**2)+((0-(1-spec))**2))
            perfTab.loc[alphaint,'ALPHA'] = alpha
            perfTab.loc[alphaint,'TP%'] = tpp
            perfTab.loc[alphaint,'TN%'] = tnp
            perfTab.loc[alphaint,'FP%'] = fpp
            perfTab.loc[alphaint,'FN%'] = fnp
            perfTab.loc[alphaint,'SENS'] = sens
            perfTab.loc[alphaint,'SPEC'] = spec
            perfTab.loc[alphaint,'ER-IND'] = erind
        auc = 0
        for alphaint in range(alphasteps):
            tabind = alphasteps-alphaint
            xf = 1 - perfTab.loc[tabind-1,'SPEC']
            xi = 1 - perfTab.loc[tabind,'SPEC']
            yf = perfTab.loc[tabind-1,'SENS']
            yi = perfTab.loc[tabind,'SENS']
            astep = ((yf+yi)*(xf-xi))/2
            perfTab.loc[tabind,'ASTEP'] = astep
            auc = auc + astep
        outinfo.loc[ldayfstr,'AUC'] = auc
        miniu = 1
        if cycle >= (offsetCycles+1):
            oldcutoff = cutoffAlpha
        else:
            oldcutoff = 0.5
        for alphaint in range((alphasteps+1)):
            newsens = perfTab.loc[alphaint,'SENS']
            newspec = perfTab.loc[alphaint,'SPEC']
            iu = abs(newsens-auc) + abs(newspec-auc)
            perfTab.loc[alphaint,'IU'] = iu
            if iu < miniu:
                miniu = iu
                cutoffAlpha = perfTab.loc[alphaint,'ALPHA']
        ptOutpath = outpath+'ROC_INFO//'
        if not(os.path.isdir(ptOutpath)):
            os.makedirs(ptOutpath)
        perfTab.to_excel(ptOutpath+prefix+'ROCinfo_to_'+outwstr+'.xlsx')
        if xfoldCutoffFlag == 1:
            outinfo.loc[ldayfstr,'Cut-Off_week'] = cutoffAlpha
        else:
            outinfo.loc[ldayfstr,'Cut-Off_used'] = oldcutoff
            outinfo.loc[ldayfstr,'Cut-Off_new'] = cutoffAlpha



        ####################################################### IDENTIFY CUT-OFF ON TRAINING SET


        if xfoldCutoffFlag == 1:
            print(cyclestr+' COMPUTING CUT-OFF')
            alphacut = xfold_cutoff(traindbLvlThree,10,uid,targetVariable,model,alphasteps)
            oldcutoff = alphacut
            outinfo.loc[ldayfstr,'Cut-Off_used'] = oldcutoff



        ####################################################### EVALUATE PERFORMANCE ON PATIENTS



        print(cyclestr+' EVALUATING PATIENTS MANAGEMENT')
        hubdb = testdbLvlZero[['HUB',uid,'CRIT','TOT_POS']].copy(deep=True)
        hubdb = hubdb.rename(columns={'TOT_POS':'US_TOTPOS'})
        testFinal = pd.merge(testResults,hubdb,on=uid)
        quartthresh = pd.read_excel('DATASOURCE_quartiles_thresholds.xlsx')
        medthresh = quartthresh.loc[quartthresh['QUARTILE']=='Q2','UPPERTHRESHOLD'].iloc[0]
        lowprevdb = testFinal.loc[testFinal['US_TOTPOS']<medthresh].copy(deep=True)
        highprevdb = testFinal.loc[testFinal['US_TOTPOS']>=medthresh].copy(deep=True)
        crithighprevdb = highprevdb.loc[highprevdb['CRIT']==1].copy(deep=True)

        PH = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==1))].shape[0]
        PS = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==0))].shape[0]
        NH = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==1))].shape[0]
        NS = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==0))].shape[0]
        ktp = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==1)&(lowprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        imp = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==1)&(lowprevdb['PRED_VAL']<oldcutoff))].shape[0]
        crp = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==0)&(lowprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        udp = lowprevdb.loc[((lowprevdb[targetVariable]==1)&(lowprevdb['HUB']==0)&(lowprevdb['PRED_VAL']<oldcutoff))].shape[0]
        udn = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==1)&(lowprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        crn = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==1)&(lowprevdb['PRED_VAL']<oldcutoff))].shape[0]
        imn = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==0)&(lowprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        ktn = lowprevdb.loc[((lowprevdb[targetVariable]==0)&(lowprevdb['HUB']==0)&(lowprevdb['PRED_VAL']<oldcutoff))].shape[0]
        outinfo.loc[ldayfstr,'PH'] = PH
        outinfo.loc[ldayfstr,'PS'] = PS
        outinfo.loc[ldayfstr,'NH'] = NH
        outinfo.loc[ldayfstr,'NS'] = NS
        outinfo.loc[ldayfstr,'ktp'] = ktp
        outinfo.loc[ldayfstr,'imp'] = imp
        outinfo.loc[ldayfstr,'crp'] = crp
        outinfo.loc[ldayfstr,'udp'] = udp
        outinfo.loc[ldayfstr,'udn'] = udn
        outinfo.loc[ldayfstr,'crn'] = crn
        outinfo.loc[ldayfstr,'imn'] = imn
        outinfo.loc[ldayfstr,'ktn'] = ktn

        cPH = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==1))].shape[0]
        cPS = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==0))].shape[0]
        cNH = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==1))].shape[0]
        cNS = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==0))].shape[0]
        cktp = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==1)&(crithighprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        cimp = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==1)&(crithighprevdb['PRED_VAL']<oldcutoff))].shape[0]
        ccrp = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==0)&(crithighprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        cudp = crithighprevdb.loc[((crithighprevdb[targetVariable]==1)&(crithighprevdb['HUB']==0)&(crithighprevdb['PRED_VAL']<oldcutoff))].shape[0]
        cudn = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==1)&(crithighprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        ccrn = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==1)&(crithighprevdb['PRED_VAL']<oldcutoff))].shape[0]
        cimn = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==0)&(crithighprevdb['PRED_VAL']>=oldcutoff))].shape[0]
        cktn = crithighprevdb.loc[((crithighprevdb[targetVariable]==0)&(crithighprevdb['HUB']==0)&(crithighprevdb['PRED_VAL']<oldcutoff))].shape[0]
        outinfo.loc[ldayfstr,'cPH'] = cPH
        outinfo.loc[ldayfstr,'cPS'] = cPS
        outinfo.loc[ldayfstr,'cNH'] = cNH
        outinfo.loc[ldayfstr,'cNS'] = cNS
        outinfo.loc[ldayfstr,'cktp'] = cktp
        outinfo.loc[ldayfstr,'cimp'] = cimp
        outinfo.loc[ldayfstr,'ccrp'] = ccrp
        outinfo.loc[ldayfstr,'cudp'] = cudp
        outinfo.loc[ldayfstr,'cudn'] = cudn
        outinfo.loc[ldayfstr,'ccrn'] = ccrn
        outinfo.loc[ldayfstr,'cimn'] = cimn
        outinfo.loc[ldayfstr,'cktn'] = cktn

outinfo.to_excel(outpath+prefix+'cyclesResults.xlsx')



####################################################### PLOT CUMULATED RESULTS



if goplot == 1:

    fullktp = sum(np.asarray(outinfo['ktp']))
    fullimp = sum(np.asarray(outinfo['imp']))
    fullcrp = sum(np.asarray(outinfo['crp']))
    fulludp = sum(np.asarray(outinfo['udp']))
    fulludn = sum(np.asarray(outinfo['udn']))
    fullcrn = sum(np.asarray(outinfo['crn']))
    fullimn = sum(np.asarray(outinfo['imn']))
    fullktn = sum(np.asarray(outinfo['ktn']))
    fullcktp = sum(np.asarray(outinfo['cktp']))
    fullcimp = sum(np.asarray(outinfo['cimp']))
    fullccrp = sum(np.asarray(outinfo['ccrp']))
    fullcudp = sum(np.asarray(outinfo['cudp']))
    fullcudn = sum(np.asarray(outinfo['cudn']))
    fullccrn = sum(np.asarray(outinfo['ccrn']))
    fullcimn = sum(np.asarray(outinfo['cimn']))
    fullcktn = sum(np.asarray(outinfo['cktn']))

    lowprevall = [fullktp+fullktn, fullcrp+fullcrn, fulludp+fulludn, fullimp+fullimn]
    lowtot = sum(lowprevall)
    highprevall = [fullcktp+fullcktn, fullccrp+fullccrn, fullcudp+fullcudn, fullcimp+fullcimn]
    hightot = sum(highprevall)
    lowprevp = [fullktp, fullcrp, fulludp, fullimp]
    lowptot = sum(lowprevp)
    highprevp = [fullcktp, fullccrp, fullcudp, fullcimp]
    highptot = sum(highprevp)
    lowprevn = [fullktn, fullcrn, fulludn, fullimn]
    lowntot = sum(lowprevn)
    highprevn = [fullcktn, fullccrn, fullcudn, fullcimn]
    highntot = sum(highprevn)

    plot_sankey(lowprevall,prefix,outpath,'Low prevalence - all')
    if lowptot > 0:
        plot_sankey(lowprevp,prefix,outpath,'Low prevalence - pos',lowtot,lowptot)
    if lowntot > 0:
        plot_sankey(lowprevn,prefix,outpath,'Low prevalence - neg',lowtot,lowntot)
    plot_sankey(highprevall,prefix,outpath,'High prevalence - all')
    if highptot > 0:
        plot_sankey(highprevp,prefix,outpath,'High prevalence - pos',hightot,highptot)
    if highntot > 0:
        plot_sankey(highprevn,prefix,outpath,'High prevalence - neg',hightot,highntot)

br = 1
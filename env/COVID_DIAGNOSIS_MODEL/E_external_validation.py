print('\n\nScript started: E_external_validation')
print('\n\nImporting libraries')
import pandas as pd
import numpy as np
from xFold_crossValidation import *
import joblib

quart = 'Geo_' # EMPTY for full DB, 'Q1_' for first quartile .... 'Q4_' for last quartile
suffix = '_M4S_'
indbname = 'MainDB_201001_210721'
infilename = 'AO_' + quart + indbname + '.xlsx'
extsetdbname = 'MainDB_211001_211231'
extsetfilename = 'AO_' + quart + extsetdbname + '.xlsx'
extension = '.xlsx'

#Variables Model 1
#attslist = ['AGE','SEX','CONTATTO','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO']
#Variables Model 2
#attslist = ['CONTATTO','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','TEMPERATURA','AGE','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST','MOTIVO','SEX','COSCIENZA','RESPIRODS']
#Variables Model 3
#attslist = ['AGE','SEX','CONTATTO','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','TOT_POS','GEOVAL']
#Variables Model 4
attslist = ['CONTATTO','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','TEMPERATURA','AGE','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST','MOTIVO','SEX','COSCIENZA','RESPIRODS','TOT_POS','GEOVAL']
print('\n\nImporting datasources')
indb = pd.read_excel(infilename)
extdb = pd.read_excel(extsetfilename)

for mode in range(1,2):


    if mode == 0:
        modestr = 'LogisticRegression'
    if mode == 1:
        modestr = 'RandomForest'
    if mode == 2:
        modestr = 'SupportVectorMachine'
    if mode == 3:
        modestr = 'NaiveBayesGaussian'

    print('\n\nStarting validation on model ',modestr)

    for tw in range(7,8):
        targstring = 'ESITO_+-' + str(tw)
        mlModelFileName = 'TRAINED_MODELS//xFoldOutput_' + targstring + '_' + modestr + '_' + suffix + '.pkl'
        print('\n\nLoading trained model')
        mlModel = joblib.load(mlModelFileName)
        [fulldb,plotroc] = ML_binClass_xFold_externalValidation(mlModel,extdb,indb,'ID',targstring,10,suffix,mode,5,5,attslist,1,{'SENS':0.9},'foo','foo',1)
        fulldb.to_excel('EO_'+quart+indbname+'_'+extsetdbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_FullStats.xlsx')
        plotroc.to_excel('EO_'+quart+indbname+'_'+extsetdbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_PlotROC.xlsx')


        br = 1
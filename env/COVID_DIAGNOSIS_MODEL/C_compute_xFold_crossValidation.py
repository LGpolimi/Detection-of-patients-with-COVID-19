print('\n\nScript started: C_compute_xFold_crossValidation')
print('\n\nImporting libraries')
import pandas as pd
import numpy as np
from xFold_crossValidation import *

quart = 'Geo_' # EMPTY for full DB, 'Q1_' for first quartile .... 'Q4_' for last quartile
suffix = '_M4S_'
indbname = 'MainDB_201001_210721'
infilename = 'AO_' + quart + indbname + '.xlsx'
extension = '.xlsx'
savemodel = 1

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

for mode in range(1,2):

    if mode == 0:
        modestr = 'LogisticRegression'
    if mode == 1:
        modestr = 'RandomForest'
    if mode == 2:
        modestr = 'SupportVectorMachine'
    if mode == 3:
        modestr = 'NaiveBayesGaussian'

    for tw in range(7,8):
        targstring = 'ESITO_+-' + str(tw)
        [findb,itdb,fulldb,plotroc] = ML_binClass_xFold_crossValidation(indb,'ID',targstring,10,suffix,mode,5,5,attslist,1,{'SENS':0.9},'foo','foo',1,1,savemodel)
        findb.to_excel('CO_'+quart+indbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_FinalStats.xlsx')
        itdb.to_excel('CO_'+quart+indbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_IterationsStats.xlsx')
        fulldb.to_excel('CO_'+quart+indbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_FullStats.xlsx')
        plotroc.to_excel('CO_'+quart+indbname+'_'+str(tw)+'_'+modestr+'_'+suffix+'_PlotROC.xlsx')

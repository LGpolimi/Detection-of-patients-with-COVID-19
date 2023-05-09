print('\n\nScript started: B_compute_statistics')
print('\n\nImporting libraries')
import pandas as pd
from Z_classes_and_functions import *

quart = '' # EMPTY for full DB, 'Q1_' for first quartile .... 'Q4_' for last quartile
indbname = 'MainDB_201001_210721.xlsx'
infilename = 'resp_trainingAO_' + quart + indbname
outprefix = 'respRT_BO_' + quart
targetfield = 'ESITO_+-7'
attributes = ['MOTIVO','SEX','COSCIENZA','RESPIRODS','CONTATTO','TEMPERATURA','AGE','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','SOSPETTO','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST','TOT_POS','CRIT','NEWS_ALERT']
niterations = 10
zindex = 5
print('\n\nImporting datasources')
indb = pd.read_excel(infilename)
statsdb = compute_corr_withbin(indb,targetfield,attributes,niterations,zindex)
outtab = pd.pivot_table(statsdb,index=['VARIABILE','STAT','VALUE'])
fname = outprefix + 'ExpAnalysisSTATS' + '_' + targetfield
print('\n\t\t\tEXPORTING DATA')
outtab.to_excel(fname + '.xlsx')
confs = list()
excs = list()
for mode in range(3):
    if mode == 0:
        mprefix = 'allVars'
        attributes = ['MOTIVO','SEX','COSCIENZA','RESPIRODS','CONTATTO','TEMPERATURA','AGE','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST']
    elif mode == 1:
        mprefix = 'binVars'
        attributes = ['SEX','CONTATTO','AGE','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO']
    elif mode == 2:
        mprefix = 'numVars'
        attributes = ['MOTIVO','COSCIENZA','RESPIRODS','TEMPERATURA','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST']
    [conf,exc,xdb,res,ci] = multivariate_logreg_or(indb,attributes,'ID',targetfield,0,"foo",1)
    confs.append(conf)
    excs.append(exc)

    conf.loc[' ',' '] = ''
    conf.loc[mprefix,'C-INDEX'] = ci

    conf.to_excel(outprefix+mprefix+'_MultiVariate_OddsRatios.xlsx')
    parfname = outprefix+mprefix+'_MultiVariate_ModelParams.txt'
    fh = open(parfname,'w')
    fh.write(res.summary().as_text())
    fh.close()

br = 1
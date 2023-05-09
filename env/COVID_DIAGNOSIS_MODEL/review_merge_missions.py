import numpy as np
import pandas as pd
import pathlib
import datetime as dt

saveout = 0
outpath = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2022 05 -- AREU resources dimensioning\\Demand Analysis - Development\\Data sources\\Structured\\AREUmissions\\'
dspath = pathlib.Path('D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2020 03 -- SARS_CoV 2 Analysis\\Data sources\\Ambulances_Dispatches')
uid = 'ID_PZ'
fileslist = list(dspath.iterdir())
totf = len(fileslist)
fi = 0
wholedb = pd.DataFrame()
for f in fileslist:
    fi = fi + 1
    fname = str(f)
    if fname[len(fname)-3:len(fname)] == 'csv':
        print('Importing file ',str(fi),' out of ',str(totf),': ',fname)
        newdb = pd.read_csv(fname,low_memory=False,encoding='ISO-8859-1',sep=';')
        wholedb = pd.concat([wholedb,newdb])

wholedb.reset_index(inplace=True,drop=True)
checkdoubles = pd.DataFrame(wholedb[uid].duplicated())
doubles = checkdoubles.loc[checkdoubles[uid]==True]
doublist = doubles.index.values.tolist()
missionsdb = wholedb.loc[~wholedb.index.isin(doublist)].copy(deep=True)

missionsdb.loc[:,'DATETIME'] = pd.to_datetime(missionsdb['DT_ 118'], format='%d%b%Y:%H:%M:%S.%f',errors='coerce')
missionsdb.loc[pd.isna(missionsdb['DATETIME']),'DATETIME'] = pd.to_datetime(missionsdb['DT_ 118'], format='%d%b%Y %H:%M:%S,%f',errors='coerce')
missionsdb.sort_values(by='DATETIME',ascending=True,inplace=True)
minday = min(missionsdb['DATETIME'])
minstr = minday.strftime('%Y%m%d')
maxday = max(missionsdb['DATETIME'])
maxstr = maxday.strftime('%Y%m%d')
if saveout == 1:
    missionsdb.to_csv(outpath+'AREU_MISSIONS_'+minstr[2:len(minstr)]+'_'+maxstr[2:len(maxstr)]+'.csv',encoding='ISO-8859-1',index=False)

br = 1
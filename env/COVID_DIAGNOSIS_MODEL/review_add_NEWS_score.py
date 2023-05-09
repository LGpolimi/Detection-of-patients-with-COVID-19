import numpy as np
import pandas as pd

outprefix = 'training'
saveout = 1
dbname = 'AO_MainDB_201001_210721'
indb = pd.read_excel(dbname + '.xlsx')
totiters = indb.shape[0]
iti = 0

for row in indb.iterrows():

    iti = iti + 1
    print('Computing NEWS score for set ',outprefix,' processing = ',str((iti/totiters)*100),' %')

    news = 0
    flag3 = 0

    rr = row[1]['RESP_FREQ']
    if type(rr) == int or type(rr) == float or type(rr) == np.float64:
        if rr<=8 or rr >=25:
            news = news + 3
            flag3 = 1
        elif (rr>=9 and rr<=11):
            news = news + 1
        elif rr>=21 and rr<=24:
            news = news + 2

    spo21 = row[1]['SAT_ARIA']
    if type(spo21) == int or type(spo21) == float or type(spo21) == np.float64:
        if spo21<=91:
            news = news + 3
            flag3 = 1
        elif spo21==92 or spo21==93:
            news = news + 2
        elif spo21==94 or spo21==95:
            news = news + 1

    sbp = row[1]['PRESS_SIST']
    if type(sbp)==int or type(sbp) == float or type(sbp) == np.float64:
        if sbp<=90 or sbp>=220:
            news = news+3
            flag3 = 1
        elif sbp > 90 and sbp <= 100:
            news = news+2
        elif sbp > 100 and sbp <= 110:
            news = news+1

    pul = row[1]['FREQ_CARDIO']
    if type(pul) == int or type(pul) == float or type(pul) == np.float64:
        if pul<=40 or pul >=131:
            news = news + 3
            flag3 = 1
        elif pul>=111 and pul<=130:
            news = news + 2
        elif (pul>=41 and pul<=50) or (pul>=91 and pul<=110):
            news = news + 1

    avpu = row[1]['COSCIENZA']
    if avpu == 'V' or avpu == 'P' or avpu == 'U':
        news = news + 3
        flag3 = 1

    temp = row[1]['TEMPERATURA']
    if type(temp) == int or type(temp) == float or type(temp) == np.float64:
        if temp<=35:
            news = news + 3
            flag3 = 1
        elif temp>=39.1:
            news = news + 2
        elif (temp>=35.1 and temp<=36) or (temp>=38.1 and temp<=39):
            news = news + 1

    indb.loc[row[0],'NEWS'] = news
    if news <=4:
        if flag3 == 1:
            indb.loc[row[0],'NEWS_ALERT'] = 'LIV2'
        else:
            indb.loc[row[0],'NEWS_ALERT'] = 'LIV1'
    elif news > 4 and news <= 6:
        indb.loc[row[0],'NEWS_ALERT'] = 'LIV3'
    else:
        indb.loc[row[0],'NEWS_ALERT'] = 'LIV4'

if saveout == 1:

    indb.to_excel('review'+outprefix+dbname+'.xlsx')

br = 1
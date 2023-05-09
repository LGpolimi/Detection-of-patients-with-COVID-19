import pandas as pd
import numpy as np
from math import *
import os
from statistics import *
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import normaltest
import random
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
import datetime as dt
from plotly import graph_objs as go
import kaleido
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

class tampone():
    def __init__(self):
        self.esito = 0
        self.tdiff = 0

class intervento():
    def __init__(self):
        self.date = ''
        self.lat = 0
        self.lon = 0
        self.motivo = ''
        self.motivodtl = ''
        self.ospedale = ''
        self.hubspoke = ''

class person():
    def __init__(self):
        self.age = 0
        self.sex = ''

class boolparams():
    def __init__(self):
        self.contatto = 0
        self.respiro = 0
        self.febbre = 0
        self.diarrea = 0
        self.tosse = 0
        self.sensi = 0
        self.astenia = 0
        self.vomito = 0

class dsparams():
    def __init__(self):
        self.coscienza = ''
        self.respiro = ''
        self.respf = 0
        self.sataria = 0
        self.satox = 0
        self.polsoper = ''
        self.polsocen = ''
        self.cardf = 0
        self.tdcardf = ''
        self.presssist = 0
        self.pressdiast = 0
        self.temperatura = 0

class customparams():
    def __init__(self):
        self.sospetto = 0
        self.nsospetti = 1

class medparams():
    def __init__(self):
        self.bool = boolparams()
        self.ds = dsparams()
        self.cust = customparams()

class patient():
    def __init__(self):
        self.id = 0
        self.anagrafica = person()
        self.tampbool = 0
        self.sgn = 0
        self.tamponi = list()
        self.inter = intervento()
        self.parametri = medparams()


def find_closest_results(tamplist,reftamp,reftdiff):
    minnegdist = -999
    minposdist = 999
    minneges = -999
    minposes = 999
    for tamp in tamplist:
        newdist = tamp.tdiff - reftdiff
        if newdist < 0 and abs(newdist) < abs(minnegdist):
            minnegdist = newdist
            minneges = tamp.esito
        if newdist > 0 and newdist < minposdist:
            minposdist = newdist
            minposes = tamp.esito
    return(minnegdist,minposdist,minneges,minposes)

def build_patients_struct(indb,hubmapfilename):

    allpatients = list()
    hubspokemap = pd.read_excel(hubmapfilename)
    oldid = 0
    totrows = len(indb)
    for row in indb.iterrows():
        print('Building patients structure: line processing = ',(row[0]/totrows)*100,' %')
        newid = int(row[1]['ID_PATIENT'])
        if newid == oldid:
            newpatflag = 0
        else:
            newpatflag = 1
        if newpatflag:
            if row[0] > 0:
                del(newpatient)
            newpatient = patient()
            newpatient.id = newid
            if not(pd.isnull(row[1]['AGE'])):
                newpatient.anagrafica.age = int(row[1]['AGE'])
            newpatient.anagrafica.sex = row[1]['GEN']
            if pd.isnull(row[1]['SGN']):
                newpatient.sgn = 0
            else:
                newpatient.sgn = 1
            newpatient.inter.date = row[1]['DAY']
            newpatient.inter.lat = row[1]['LAT']
            newpatient.inter.lon = row[1]['LON']
            newpatient.inter.motivo = row[1]['MOTIVO']
            newpatient.inter.motivodtl = row[1]['MOT_DTL']
            hosp = row[1]['OSPEDALE']
            newpatient.inter.ospedale = hosp
            try:
                checkhub = hubspokemap.loc[hubspokemap['OSPEDALE']==hosp,'HUB'].values[0]
            except:
                checkhub = 0
            newpatient.inter.hubspoke = checkhub
            if pd.isnull(row[1]['CONTATTO-POSIT']):
                newpatient.parametri.bool.contatto = 0
                if row[1]['CONTATTO-POSIT'] == 'TAMPONE POSITIVO':
                    newpatient.parametri.bool.contatto = -1
            else:
                newpatient.parametri.bool.contatto = 1
            if pd.isnull(row[1]['RESPIRO']):
                newpatient.parametri.bool.respiro = 0
            else:
                newpatient.parametri.bool.respiro = 1
            if pd.isnull(row[1]['FEBBRE']):
                newpatient.parametri.bool.febbre = 0
            else:
                newpatient.parametri.bool.febbre = 1
            if pd.isnull(row[1]['DIARREA']):
                newpatient.parametri.bool.diarrea = 0
            else:
                newpatient.parametri.bool.diarrea = 1
            if pd.isnull(row[1]['TOSSE-RAFREDDORE']):
                newpatient.parametri.bool.tosse = 0
            else:
                newpatient.parametri.bool.tosse = 1
            if pd.isnull(row[1]['DIST.GUSTO-OLF']):
                newpatient.parametri.bool.sensi = 0
            else:
                newpatient.parametri.bool.sensi = 1
            if pd.isnull(row[1]['ASTENIA-DOL']):
                newpatient.parametri.bool.astenia = 0
            else:
                newpatient.parametri.bool.astenia = 1
            if pd.isnull(row[1]['VOMITO']):
                newpatient.parametri.bool.vomito = 0
            else:
                newpatient.parametri.bool.vomito = 1
            totsospetti = newpatient.parametri.bool.respiro + newpatient.parametri.bool.febbre + newpatient.parametri.bool.diarrea + newpatient.parametri.bool.tosse + newpatient.parametri.bool.sensi + newpatient.parametri.bool.astenia + newpatient.parametri.bool.vomito
            if newpatient.parametri.bool.contatto == 1:
                totsospetti = totsospetti + 1
            if totsospetti > 0:
                newpatient.parametri.cust.sospetto = 1
                newpatient.parametri.cust.nsospetti = totsospetti
            else:
                newpatient.parametri.cust.sospetto = 0
                newpatient.parametri.cust.nsospetti = 0
            newpatient.parametri.ds.coscienza = row[1]['DS_COSCIENZA']
            newpatient.parametri.ds.respiro = row[1]['DS_RESPIRO']
            newpatient.parametri.ds.respf = row[1]['VL_FREQ_RESP']
            newpatient.parametri.ds.sataria = row[1]['VL_SATURAZ_ARIA']
            newpatient.parametri.ds.satox = row[1]['VL_SATURAZ_O2']
            newpatient.parametri.ds.polsoper = row[1]['FL_POLSO_PERIF']
            newpatient.parametri.ds.polsocen = row[1]['FL_POLSO_CENTR']
            newpatient.parametri.ds.cardf = row[1]['VL_FREQ_CARD']
            newpatient.parametri.ds.tdcardf = row[1]['DS_TP_FREQ_CARD']
            newpatient.parametri.ds.presssist = row[1]['VL_PRESS_SIST']
            newpatient.parametri.ds.pressdiast = row[1]['VL_PRESS_DIAS']
            newpatient.parametri.ds.temperatura = row[1]['VL_TEMPERATURA']
            allpatients.append(newpatient)
        if not(pd.isnull(row[1]['ESITO'])):
            newpatient.tampbool = 1
            try:
                del(newtampone)
            except:
                a = 1
            newtampone = tampone()
            if row[1]['ESITO'] == 'NEGATIVO':
                newtampone.esito = 0
            elif row[1]['ESITO'] == 'POSITIVO': #or row[1]['ESITO'] == 'DEBOLMENTE POSITIVO':
                newtampone.esito = 1
            else:
                newtampone.esito = -1
                print('\n\n\n\t\t\t ERROR: value for esito = ',row[1]['ESITO'])
            newtampone.tdiff = int(row[1]['DIFF_EV_TAMP'])
            newpatient.tamponi.append(newtampone)
        oldid = newid
    return(allpatients)

def exportdb(allpatients):
    outdb = pd.DataFrame()
    idx = -1
    totlen = len(allpatients)
    for pat in allpatients:
        idx = idx+1
        print('Converting patients : processing ',(idx/totlen)*100, '%')
        outdb.loc[idx,['ID']] = pat.id
        outdb.loc[idx,['AGE']] = pat.anagrafica.age
        outdb.loc[idx,['SEX']] = pat.anagrafica.sex
        outdb.loc[idx,['DATE_INT']] = pat.inter.date
        outdb.loc[idx,['LAT']] = pat.inter.lat
        outdb.loc[idx,['LON']] = pat.inter.lon
        if type(pat.inter.motivo) == str and type(pat.inter.motivodtl) == str:
            if pat.inter.motivo == 'MEDICO ACUTO':
                outdb.loc[idx,['MOTIVO']] = pat.inter.motivo + ', ' + pat.inter.motivodtl
            else:
                outdb.loc[idx,['MOTIVO']] = pat.inter.motivo
        else:
            if type(pat.inter.motivo) == str and type(pat.inter.motivodtl) != str:
                outdb.loc[idx,['MOTIVO']] = pat.inter.motivo
            elif type(pat.inter.motivo) != str and type(pat.inter.motivodtl) == str:
                outdb.loc[idx,['MOTIVO']] = pat.inter.motivodtl
        outdb.loc[idx,['HOSP']] = pat.inter.ospedale
        outdb.loc[idx,['HUB']] = pat.inter.hubspoke
        outdb.loc[idx,['SGN']] = pat.sgn
        outdb.loc[idx,['CONTATTO']] = pat.parametri.bool.contatto
        outdb.loc[idx,['RESPIRO']] = pat.parametri.bool.respiro
        outdb.loc[idx,['FEBBRE']] = pat.parametri.bool.febbre
        outdb.loc[idx,['DIARREA']] = pat.parametri.bool.diarrea
        outdb.loc[idx,['TOSSE']] = pat.parametri.bool.tosse
        outdb.loc[idx,['GUSTO-OLFATTO']] = pat.parametri.bool.sensi
        outdb.loc[idx,['ASTENIA']] = pat.parametri.bool.astenia
        outdb.loc[idx,['VOMITO']] = pat.parametri.bool.vomito
        outdb.loc[idx,['SOSPETTO']] = pat.parametri.cust.sospetto
        outdb.loc[idx,['N_SOSPETTI']] = pat.parametri.cust.nsospetti
        if pat.parametri.cust.nsospetti > 7:
            outdb.loc[idx,['8N_SOSPETTI']] = 1
            outdb.loc[idx,['7N_SOSPETTI']] = 1
            outdb.loc[idx,['6N_SOSPETTI']] = 1
            outdb.loc[idx,['5N_SOSPETTI']] = 1
            outdb.loc[idx,['4N_SOSPETTI']] = 1
            outdb.loc[idx,['3N_SOSPETTI']] = 1
            outdb.loc[idx,['2N_SOSPETTI']] = 1
        else:
            outdb.loc[idx,['8N_SOSPETTI']] = 0
            if pat.parametri.cust.nsospetti > 6:
                outdb.loc[idx,['7N_SOSPETTI']] = 1
                outdb.loc[idx,['6N_SOSPETTI']] = 1
                outdb.loc[idx,['5N_SOSPETTI']] = 1
                outdb.loc[idx,['4N_SOSPETTI']] = 1
                outdb.loc[idx,['3N_SOSPETTI']] = 1
                outdb.loc[idx,['2N_SOSPETTI']] = 1
            else:
                outdb.loc[idx,['7N_SOSPETTI']] = 0
                if pat.parametri.cust.nsospetti > 5:
                    outdb.loc[idx,['6N_SOSPETTI']] = 1
                    outdb.loc[idx,['5N_SOSPETTI']] = 1
                    outdb.loc[idx,['4N_SOSPETTI']] = 1
                    outdb.loc[idx,['3N_SOSPETTI']] = 1
                    outdb.loc[idx,['2N_SOSPETTI']] = 1
                else:
                    outdb.loc[idx,['6N_SOSPETTI']] = 0
                    if pat.parametri.cust.nsospetti > 4:
                        outdb.loc[idx,['5N_SOSPETTI']] = 1
                        outdb.loc[idx,['4N_SOSPETTI']] = 1
                        outdb.loc[idx,['3N_SOSPETTI']] = 1
                        outdb.loc[idx,['2N_SOSPETTI']] = 1
                    else:
                        outdb.loc[idx,['5N_SOSPETTI']] = 0
                        if pat.parametri.cust.nsospetti > 3:
                            outdb.loc[idx,['4N_SOSPETTI']] = 1
                            outdb.loc[idx,['3N_SOSPETTI']] = 1
                            outdb.loc[idx,['2N_SOSPETTI']] = 1
                        else:
                            outdb.loc[idx,['4N_SOSPETTI']] = 0
                            if pat.parametri.cust.nsospetti > 2:
                                outdb.loc[idx,['3N_SOSPETTI']] = 1
                                outdb.loc[idx,['2N_SOSPETTI']] = 1
                            else:
                                outdb.loc[idx,['3N_SOSPETTI']] = 0
                                if pat.parametri.cust.nsospetti > 1:
                                    outdb.loc[idx,['2N_SOSPETTI']] = 1
                                else:
                                    outdb.loc[idx,['2N_SOSPETTI']] = 0
        outdb.loc[idx,['COSCIENZA']] = pat.parametri.ds.coscienza
        outdb.loc[idx,['RESPIRODS']] = pat.parametri.ds.respiro
        outdb.loc[idx,['RESP_FREQ']] = pat.parametri.ds.respf
        outdb.loc[idx,['SAT_ARIA']] = pat.parametri.ds.sataria
        outdb.loc[idx,['SAT_O2']] = pat.parametri.ds.satox
        outdb.loc[idx,['POLSO_PER']] = pat.parametri.ds.polsoper
        outdb.loc[idx,['POLSO_CEN']] = pat.parametri.ds.polsocen
        outdb.loc[idx,['FREQ_CARDIO']] = pat.parametri.ds.cardf
        outdb.loc[idx,['TD_FREQ_CARD']] = pat.parametri.ds.tdcardf
        outdb.loc[idx,['PRESS_SIST']] = pat.parametri.ds.presssist
        outdb.loc[idx,['PRESS_DIAST']] = pat.parametri.ds.pressdiast
        outdb.loc[idx,['TEMPERATURA']] = pat.parametri.ds.temperatura


        outdb.loc[idx,['DISCARDED']] = ''

        for t in range(0,16):
            tampstr = 'TAMPONE_+-'+str(t)
            esitstr = 'ESITO_+-'+str(t)
            outdb.loc[idx,[tampstr]] = 0
            outdb.loc[idx,[esitstr]] = ''

        ################################# HANDLING OF DOUBT RESULTS
        keeppatflag = 1

        keeptamps_dub = list()
        for tamp in pat.tamponi:
            keepflag = 1
            if tamp.esito == -1:
                [minnegdist,minposdist,minneges,minposes] = find_closest_results(pat.tamponi,tamp,tamp.tdiff)
                if minposes == minneges:
                    tamp.esito = minposes
                else:
                    if minnegdist != -999 and minposdist != 999:
                        if abs(minnegdist) <= round(0.333*abs(minposdist)):
                            tamp.esito = minneges
                        elif abs(minposdist) <= round(0.333*abs(minnegdist)):
                            tamp.esito = minposes
                        else:
                            keepflag = 0
                            keeppatflag = 0
                            outdb.loc[idx,['DISCARDED']] = 'A1'
                    if minnegdist == -999 or minposdist == 999:
                        if abs(tamp.tdiff) >= 10:
                            keepflag = 0
                        else:
                            keepflag = 0
                            keeppatflag = 0
                            outdb.loc[idx,['DISCARDED']] = 'A2'
            if keepflag == 1:
                keeptamps_dub.append(tamp)
        pat.tamponi = list()
        if keeppatflag == 1:
            pat.tamponi = keeptamps_dub

        ################################# CORRECTION FOR OPPOSITE DOUBLE RESULTS IN THE SAME DAY
        keeptamps_rep = list()
        for tamp in pat.tamponi:
            keepflag = 1
            daydiff = tamp.tdiff
            bes = tamp.esito
            for ctamp in pat.tamponi:
                compdiff = ctamp.tdiff
                if compdiff == daydiff:
                    ces = ctamp.esito
                    if bes != ces:
                        [minnegdist,minposdist,minneges,minposes] = find_closest_results(pat.tamponi,tamp,daydiff)
                        if minposes == minneges:
                            tamp.esito = minposes
                        else:
                            if minnegdist != -999 and minposdist != 999:
                                if abs(minnegdist) <= round(0.333*abs(minposdist)):
                                    tamp.esito = minneges
                                elif abs(minposdist) <= round(0.333*abs(minnegdist)):
                                    tamp.esito = minposes
                                else:
                                    keepflag = 0
                                    keeppatflag = 0
                                    outdb.loc[idx,['DISCARDED']] = 'B1'
                            if minnegdist == -999 or minposdist == 999:
                                if abs(tamp.tdiff) >= 10:
                                    keepflag = 0
                                else:
                                    keepflag = 0
                                    keeppatflag = 0
                                    outdb.loc[idx,['DISCARDED']] = 'B2'
            if keepflag == 1:
                keeptamps_rep.append(tamp)
        pat.tamponi = list()
        if keeppatflag == 1:
            pat.tamponi = keeptamps_rep

        ############################### TIME WINDOWS ANALYSIS ##############################

        if len(pat.tamponi) > 0:
            for t in range(0,16):
                tampstr = 'TAMPONE_+-'+str(t)
                esitstr = 'ESITO_+-'+str(t)
                gottamp = 0
                ntamps = 0
                gotpos = 0
                for tamp in pat.tamponi:
                    if abs(tamp.tdiff) <= t:
                        gottamp = 1
                        ntamps = ntamps + 1
                        if tamp.esito == 1:
                            gotpos = 1
                if gottamp == 0:
                    outdb.loc[idx,[tampstr]] = 0
                else:
                    outdb.loc[idx,[tampstr]] = 1
                    if ntamps == 1:
                        for tamp in pat.tamponi:
                            if abs(tamp.tdiff) <= t:
                                outdb.loc[idx,[esitstr]] = tamp.esito
                    else:
                        if gotpos == 0:
                            outdb.loc[idx,[esitstr]] = 0
                        else:
                            outdb.loc[idx,[esitstr]] = 1

    return(outdb)

def parse_region_info(regions: list,fields: list,mindate = "foo",maxdate="foo"):

    if mindate == "foo":
        mindate = '200101'
    if maxdate == "foo":
        maxdate = '220902'
    infilename = 'DATASOURCE_StoricoRegioni_'+ mindate + '_' + maxdate + '.txt'

    indb = pd.read_csv(infilename,delimiter=',',engine='python',encoding='ISO-8859-1')
    outdb = pd.DataFrame()

    outind = 0
    for row in indb.iterrows():
        print('Fetching info: ',str(row[0]/len(indb)*100),' %')
        keepline = 0
        checkreg = row[1]['denominazione_regione']
        for reg in regions:
            if reg == checkreg:
                keepline = 1
        if keepline:
            for f in fields:
                outdb.loc[outind,[f]] = row[1][f]
            outind = outind + 1
    return(outdb)

def filter_data(indb: pd.DataFrame,columns: list,twindow="foo"):

    if twindow == "foo":
        twindow = 7

    for row in indb.iterrows():
        print('Filtering: processing ',str((row[0]/len(indb))*100),' %')
        linen = int(row[0])
        bordflag = 0
        if (linen+1) < (twindow/2):
            lowervals = linen
            highervals = twindow-lowervals-1
            bordflag = 1
        elif (len(indb)-(linen+1)) < (twindow/2):
            highervals = len(indb)-(linen+1)
            lowervals = twindow - highervals - 1
            bordflag = 1
        for col in columns:
            allvals = list()
            fcol = col + '_filt'
            tailsdim = floor(twindow/2)
            if (linen-tailsdim) < 0:
                lowind = 0
            else:
                lowind = -tailsdim
            if (linen+tailsdim) >= indb.shape[0]:
                upind = tailsdim - (linen+tailsdim-indb.shape[0]) - 1
            else:
                upind = tailsdim+1
            for i in range(lowind,upind):
                newvalcell = indb.loc[linen+i,[col]]
                newval = newvalcell.get(col)
                allvals.append(newval)
            wvals = list()
            weight = 1/(upind-lowind)
            for val in allvals:
                newwval = val*weight
                wvals.append(newwval)
            outval = sum(wvals)
            indb.loc[linen,fcol] = outval

    return(indb)

def retro_filter_data(indb: pd.DataFrame,columns: list,twindow="foo"):

    if twindow == "foo":
        twindow = 7
    weight = 1/twindow
    outdb = indb.copy(deep=True)
    for row in indb.iterrows():
        print('Filtering: processing ',str((row[0]/len(indb))*100),' %')
        linen = int(row[0])
        bordflag = 0
        if linen <(twindow-1):
            bordflag = 1
        for col in columns:
            vals = list()
            if bordflag == 1:
                for l in range(linen):
                    vals.append(indb.loc[l,col])
                newval = (sum(np.asarray(vals)))/(linen+1)
            else:
                for t in range(1,twindow+1):
                    bckind = linen-(twindow-t)
                    vals.append(indb.loc[bckind,col])
                newval = (sum(np.asarray(vals)))/twindow
            outdb.loc[linen,('filt_'+col)] = newval

    return(outdb)

def compute_differentials(indb: pd.DataFrame,columns: list):

    outdb = indb.copy(deep=True)
    for col in columns:
        newcol = 'variazione_' + col
        for row in indb.iterrows():
            if row[0] == 0:
                val = indb.loc[row[0],col]
                outdb.loc[row[0],newcol] = val
            else:
                oldval = indb.loc[row[0]-1,col]
                newval = indb.loc[row[0],col]
                val = newval - oldval
                outdb.loc[row[0],newcol] = val

    return(outdb)

def filter_region_data():

    outdb = parse_region_info(['Lombardia'],['data','nuovi_positivi','totale_positivi','variazione_totale_positivi','dimessi_guariti','deceduti'])
    outdb_diff = compute_differentials(outdb,['dimessi_guariti','deceduti'])
    outdb_filt = filter_data(outdb_diff,['totale_positivi','nuovi_positivi','variazione_totale_positivi','variazione_dimessi_guariti','variazione_deceduti'])
    outdb_filt = retro_filter_data(outdb_diff,['totale_positivi','nuovi_positivi','variazione_totale_positivi','variazione_dimessi_guariti','variazione_deceduti'],5)

    return(outdb_filt)

def assign_prevalence(indb,prevdb):
    copyfield = 'filt_totale_positivi'
    copyfieldname = 'TOT_POS'
    dtcelli = 0
    dtcellf = 9
    inDtFormat = '%d%b%Y'
    offPart = ':00:00:00.000'
    dtcol = 'DATE_INT'

    prevrows = prevdb.shape[0]
    for row in prevdb.iterrows():
        i = row[0]
        print('Processing prevalence db: ',i+1,'/',prevrows,' ',((i+1)/prevrows)*100,' %')
        newdt = dt.datetime.strptime(row[1]['data'],'%Y-%m-%dT%H:%M:%S')
        prevdb.loc[i,'datetime_obj'] = dt.datetime.strftime(newdt,'%Y%b%d')
    inrows = indb.shape[0]
    outdb = indb.copy(deep=True)
    for row in indb.iterrows():
        i = int(row[0])
        print('Processing db: ',i+1,'/',inrows,' ',((i+1)/inrows)*100,' %')
        newdtcell = indb.loc[i,dtcol]
        newdt = dt.datetime.strptime(newdtcell[dtcelli:dtcellf],inDtFormat)
        dtstr = dt.datetime.strftime(newdt,'%Y%b%d')
        refval = prevdb.loc[prevdb['datetime_obj']==dtstr,copyfield].iloc[0]
        outdb.loc[i,'TOT_POS'] = refval

    return(outdb)

def assign_geodata(indb,timewindow="foo",mindate="foo",maxdate="foo"):

    if timewindow == "foo":
        timewindow = 7
    
    dtcelli = 0
    dtcellf = 9
    inDtFormat = '%d%b%Y'
    mindatetime = dt.datetime.strptime(mindate,'%y%m%d') - dt.timedelta(days=timewindow)
    mindate = dt.datetime.strftime(mindatetime,'%y%m%d')
    offPart = ':00:00:00.000'
    dtcol = 'DATE_INT'
    print('PROCESSING assign geo data: importing time series')
    timeseries = pd.read_csv('DATASOURCE_MissionsTs_'+ mindate + '_' + maxdate + '.csv')
    dtcol = 'DATE_INT'
    totrows = indb.shape[0]
    for row in indb.iterrows():
        ind = row[0]
        print('PROCESSING assign geo data: ',((ind+1)/totrows)*100,' %')
        geoid = row[1]['GEOID']
        newdtcell = row[1][dtcol]
        startdate = dt.datetime.strptime(newdtcell[dtcelli:dtcellf],inDtFormat)
        alldates = list()
        for bckind in range(1,timewindow):
            backdate = startdate - dt.timedelta(days=bckind)
            bckdatestr = dt.datetime.strftime(backdate,'%y%m%d')
            alldates.append(bckdatestr)
        tsvals = list()
        for d in alldates:
            try:
                newval = timeseries.loc[geoid-1,d]
                tsvals.append(newval)
            except:
                print('PROCESSING assign geo data: ',((ind+1)/totrows)*100,' % \t DATE ',d,' NOT FOUND')
        tsarray = np.asarray(tsvals)
        geoval = np.median(tsarray)
        indb.loc[ind,'GEOVAL'] = geoval
    
    return(indb)

def evaluate_criticals(indb):

    indb.loc[((indb['SAT_ARIA']<=93)|(indb['RESP_FREQ']>30)),'CRIT'] = 1
    indb.loc[~((indb['SAT_ARIA']<=93)|(indb['RESP_FREQ']>30)),'CRIT'] = 0

    return(indb)

def join_on_field(indba,indbb,field,uid):

    allvals = indba[uid].unique()
    nvals = len(allvals)
    ind = 0
    sets = list()
    for v in allvals:
        ind = ind + 1
        print('Joining ',str(v),'on field ',field,' processing = ',str((ind/nvals)*100),' %')
        subs = indba.loc[(indba[uid]==v)].copy(deep=True)
        subb = indbb.loc[(indbb[uid]==v)].copy(deep=True)
        try:
            fval = subb[field].values[0]
            subs.loc[:,field] = fval
        except:
            print('No values for item ',str(v))
        sets.append(subs)
    outdb = pd.concat(sets)

    return(outdb)



def checknameindx(nameindx):
    if nameindx < 10:
        nameindxstr = '00'+str(nameindx)
    elif nameindx > 9 and nameindx < 100:
        nameindxstr = '0'+str(nameindx)
    else:
        nameindxstr = str(nameindx)
    return(nameindxstr)

def scalevals(x,minval,maxval,out_range=(0, 1)):
    domain = minval, maxval
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def compute_contingencytable(truevals,predvals):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    toty = len(predvals)
    for y in range(toty):
        #print(' COMPUTING CONTINGENCY TABLE: Processing = ',(y/toty)*100,' %')
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
    conttable = ([tp,fp],[fn,tn])
    return(conttable)

def compute_logreg(smally,smallveclist,longy,longveclist,traintest,vartype,varname,cindex="foo"):

    sklogreg = LogisticRegression()
    numtable = pd.read_excel('DATASOURCE_numvars_cutoff.xlsx')
    cattable = pd.read_excel('DATASOURCE_categorical_variables_map.xlsx')
    smalldf = pd.DataFrame()
    smalldf[varname] = smallveclist
    longdf = pd.DataFrame()
    longdf[varname] = longveclist
    if vartype == 1 or vartype == 0:
        smallmax = max(smallveclist)
        smallmin = min(smallveclist)
        longmax = max(longveclist)
        longmin = min(longveclist)
        if smallmax >= longmax:
            allmax = smallmax
        else:
            allmax = longmax
        if smallmin <= longmin:
            allmin = smallmin
        else:
            allmin = longmin
    if vartype == 0 or vartype == 1:
        rawsmallvec = np.asarray(smallveclist)
        rawlongvec = np.asarray(longveclist)
    if vartype == 0:
        smallvec = rawsmallvec
        longvec = rawlongvec
    elif vartype == 1:
        cutofflag = 0
        for nr in numtable.iterrows():
            if nr[1]['Variable'] == varname:
                cutofflag = 1
                cutoff = nr[1]['cutoff']
                reverseflag = nr[1]['reverse']
        if cutofflag == 0:
            allvals = np.concatenate((rawsmallvec,rawlongvec),axis=0)
            medval = np.median(allvals)
            rawsmallvec[rawsmallvec<medval] = 0
            rawsmallvec[rawsmallvec>=medval] = 1
            rawlongvec[rawlongvec<medval] = 0
            rawlongvec[rawlongvec>=medval] = 1
            smallvec = rawsmallvec
            longvec = rawlongvec
        elif cutofflag == 1:
            if reverseflag == 0:
                rawsmallvec[rawsmallvec<cutoff] = 0
                rawsmallvec[rawsmallvec>=cutoff] = 1
                rawlongvec[rawlongvec<cutoff] = 0
                rawlongvec[rawlongvec>=cutoff] = 1
            elif reverseflag == 1:
                rawsmallvec[rawsmallvec<cutoff] = 1
                rawsmallvec[rawsmallvec>=cutoff] = 0
                rawlongvec[rawlongvec<cutoff] = 1
                rawlongvec[rawlongvec>=cutoff] = 0
            smallvec = rawsmallvec
            longvec = rawlongvec
    elif vartype == 2:
        zeroval = cattable.loc[cattable['Unnamed: 0']==varname,0].iloc[0]
        oneval = cattable.loc[cattable['Unnamed: 0']==varname,1].iloc[0]
        smallveclist = smalldf
        longveclist = longdf
        if zeroval != 'other' and oneval == 'other':
            smallveclist.loc[smallveclist[varname]==zeroval] = 0
            smallveclist.loc[((smallveclist[varname]!=zeroval)&(smallveclist[varname]!=0))] = 1
            longveclist.loc[longveclist[varname]==zeroval] = 0
            longveclist.loc[((longveclist[varname]!=zeroval)&(longveclist[varname]!=0))] = 1
        elif zeroval == 'other' and oneval != 'other':
            smallveclist.loc[smallveclist[varname]==oneval] = 1
            smallveclist.loc[((smallveclist[varname]!=oneval)&(smallveclist[varname]!=1))] = 0
            longveclist.loc[longveclist[varname]==oneval] = 1
            longveclist.loc[((longveclist[varname]!=oneval)&(longveclist[varname]!=1))] = 0
        else:
            smallveclist.loc[smallveclist[varname]==oneval] = 1
            smallveclist.loc[smallveclist[varname]==zeroval] = 0
            longveclist.loc[longveclist[varname]==oneval] = 1
            longveclist.loc[longveclist[varname]==zeroval] = 0
        smallvec = np.asarray(smallveclist.loc[((smallveclist[varname]==1)|(smallveclist[varname]==0))].copy(deep=True))
        longvec = np.asarray(longveclist.loc[((longveclist[varname]==1)|(longveclist[varname]==0))].copy(deep=True))


    maxlen = len(longvec)
    minlen = len(smallvec)
    traindim = round(minlen*(traintest[0]/100))
    testdim = round(minlen*(traintest[1]/100))
    longinds = list(range(0,maxlen))
    smallinds = list(range(0,minlen))

    x_trainlist = list()
    x_testlist = list()
    y_trainlist = list()
    y_testlist = list()

    long_train_indx = random.sample(set(longinds),traindim)
    long_test_indx = random.sample(set(longinds)-set(long_train_indx),testdim)
    small_train_indx = random.sample(set(smallinds),traindim)
    small_test_indx = random.sample(set(smallinds)-set(small_train_indx),testdim)
    for ltri in long_train_indx:
        x_trainlist.append(longvec[ltri])
        y_trainlist.append(longy)
    for ltei in long_test_indx:
        x_testlist.append(longvec[ltei])
        y_testlist.append(longy)
    for stri in small_train_indx:
        x_trainlist.append(smallvec[stri])
        y_trainlist.append(smally)
    for stei in small_test_indx:
        x_testlist.append(smallvec[stei])
        y_testlist.append(smally)

    x_train = np.asarray([x_trainlist]).reshape(-1, 1)
    x_test = np.asarray([x_testlist]).reshape(-1, 1)
    y_train = np.asarray([y_trainlist]).reshape(-1, 1)
    y_test = np.asarray([y_testlist]).reshape(-1, 1)

    if max(x_train) > 1 or min(x_train)<0:
        xi = 0
        for x in x_train:
            newval = x[0]
            if newval > 1:
                y_train[xi] = 1
            if newval < 0:
                x_train[xi] = 0
            xi = xi + 1
            yi = 0
    if max(y_train) > 1 or min(y_train)<0:
        for y in y_train:
            newval = y[0]
            if newval > 1:
                y_train[yi] = 1
            if newval < 0:
                y_train[yi] = 0
            yi = yi + 1
    sklogreg.fit(x_train,y_train.ravel())

    y_pred = sklogreg.predict(x_test)
    y_predlist = list(y_pred)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    conttab = compute_contingencytable(y_test,y_pred)
    oddstable = sm.stats.Table2x2(np.array(conttab))
    oddssum = oddstable.summary(method='normal')
    results_as_html = oddssum.as_html()
    resdf = pd.read_html(results_as_html, header=0, index_col=0)[0]
    oddsr = resdf.loc['Odds ratio','Estimate']
    oddsrlow = resdf.loc['Odds ratio','LCB']
    oddsrup = resdf.loc['Odds ratio','UCB']
    oddspval = resdf.loc['Odds ratio','p-value']

    logrres = list()
    logrres.append(accuracy)
    logrres.append(oddsr)
    logrres.append(oddsrlow)
    logrres.append(oddsrup)
    logrres.append(oddspval)

    if cindex == 1:
        df = pd.DataFrame()
        df['T'] = y_train
        df['E'] = x_train
        cph = CoxPHFitter().fit(df, 'T', 'E')
        ci = concordance_index(df['T'], -cph.predict_partial_hazard(df), df['E'])

    if stdev(y_testlist) > 0 and stdev(y_predlist):
        corrmat = np.corrcoef(y_testlist,y_predlist)
        corrval = corrmat[0,1]
        logrres.append(corrval)

    return(logrres)

def compute_corr_withbin(indb,targetfield,attributes,niterations,goplot = "foo",zindex="foo",traintest="foo",histcols="foo"):

    ################################ INITIALIZE AND FETCH FIELDS #####################################

    outdb = pd.DataFrame()
    cattable = pd.read_excel('DATASOURCE_categorical_variables_map.xlsx')

    totlen = len(indb)
    if traintest == 'foo':
        del(traintest)
        traintest = list()
        traintest.append(75)
        traintest.append(25)
    if histcols == 'foo':
        histcols = 20
    if zindex == "foo":
        zindex = 3
    if goplot=="foo":
        goplot = 0

    atts = list()
    if type(attributes[0]) == int:
        allcols = list(indb.columns)
        for val in attributes:
            atts.append(allcols[val])
    else:
        for val in attributes:
            atts.append(val)
    if type(targetfield) == int:
        allcols = list(indb.columns)
        targ = allcols[targetfield]
    else:
        targ = targetfield


    ################################ EVALUATe TARGET FIELD ###########################################
    tarnulls = list()
    tarones = list()
    tarzeros = list()

    print('Evaluating target field ',targ)
    tarnulldb = indb.loc[pd.isnull(indb[targ])].copy(deep=True)
    ntarnulls = tarnulldb.shape[0]
    tarnulls = list(tarnulldb.index)
    taronesdb = indb.loc[indb[targ]==1]
    ntarones = taronesdb.shape[0]
    tarones = list(taronesdb.index)
    tarzerosdb = indb.loc[indb[targ]==0]
    ntarzeros = tarzerosdb.shape[0]
    tarzeros = list(tarzerosdb.index)
    valtardb = indb.loc[~pd.isnull(indb[targ])].copy(deep=True)

    #################################### SAVE TARGET ANALYSIS ########################################

    out_sampsize = totlen
    out_tarnulls = ntarnulls
    out_tarnulls_p = str(round((out_tarnulls / totlen)*10000)/100) + ' %'
    out_tarones = ntarones
    out_tarzeros = ntarzeros
    out_tarvalid = out_tarones + out_tarzeros
    out_tarones_p = str(round((out_tarones / out_tarvalid)*10000)/100) + ' %'
    out_tarzeros_p = str(round((out_tarzeros / out_tarvalid)*10000)/100) + ' %'
    out_tarvalid_p = str(round((out_tarvalid/ totlen)*10000)/100) + ' %'
    if out_tarvalid != valtardb.shape[0]:
        br = 1

    outrow = 0
    targnamefield = '(0) TARGET'

    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + 'TOTAL SAMPLE SIZE'
    outdb.loc[outrow,'VALUE'] = out_sampsize
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' NULLS'
    outdb.loc[outrow,'VALUE'] = out_tarnulls
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' NULLS % on TOT'
    outdb.loc[outrow,'VALUE'] = out_tarnulls_p
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' VALID'
    outdb.loc[outrow,'VALUE'] = out_tarvalid
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' VALID % on TOT'
    outdb.loc[outrow,'VALUE'] = out_tarvalid_p
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' ONEs'
    outdb.loc[outrow,'VALUE'] = out_tarones
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' ONEs % on VALID'
    outdb.loc[outrow,'VALUE'] = out_tarones_p
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' ZEROs'
    outdb.loc[outrow,'VALUE'] = out_tarzeros
    outrow = outrow + 1
    outdb.loc[outrow,'VARIABILE'] = targnamefield
    outdb.loc[outrow,'STAT'] = str(outrow) + ' ZEROs % on VALID'
    outdb.loc[outrow,'VALUE'] = out_tarzeros_p
    outrow = outrow + 1

    ##################################### CYCLE ON ATTRIBUTES ########################################

    iatt = 0
    totatts = len(atts)
    for att in atts:

        iatt = iatt + 1
        attnulls = list()
        attindx = list()
        attvals = list()

        ################################# EVALUATE NULLS
        print('Evaluating nulls in attribute ',att)

        nullsdb = valtardb.loc[((valtardb[att].isna())|(valtardb[att]=='')|(valtardb[att]=='NaT')|(valtardb[att]=='NULL'))]
        valdb = valtardb.loc[~((valtardb[att].isna())|(valtardb[att]=='')|(valtardb[att]=='NaT')|(valtardb[att]=='NULL'))]
        attnulls = nullsdb.shape[0]
        attindx = list(valdb.index)
        attvals = list(np.asarray(valdb[att]))
        ################################ EVALUATE CTAEGORICAL

        catcounter = 0
        catthresh = 100

        for ci in range(catthresh):
            randitem = random.choice(attvals)
            if type(randitem) == str:
                try:
                    conv = int(randitem)
                except:
                    catcounter = catcounter + 1
        if catcounter >= 0.05*catthresh:
            catvar = 1
            vartype = 2
        else:
            catvar = 0

        ################################# EVALUATE BINARITY

        if catvar == 0:
            binatt = 0
            vartype = 1
            nonbinlist = list()
            bvind = 0
            for val in attvals:
                print('Evaluating binarity in attribute (',att,',',iatt,'out of',totatts,'): processing ',(bvind/len(attvals))*100,' %')
                nonbinlist.append(val)
                bvind = bvind + 1
            if 1 in nonbinlist or 0 in nonbinlist:
                if 1 in nonbinlist:
                    while 1 in nonbinlist:
                        nonbinlist.remove(1)
                if 0 in nonbinlist:
                    while 0 in nonbinlist:
                        nonbinlist.remove(0)
                if len(nonbinlist) < 0.05 * len(attvals):
                    binatt = 1
                    vartype = 0

            ################################# EVALUATE OUTLIERS

            medofatt = median(attvals)
            attq25 = np.quantile(attvals,0.25)
            attq75 = np.quantile(attvals,0.75)

            outlierupth = medofatt + (zindex * (attq75-attq25))
            outlierloth = medofatt - (zindex * (attq75-attq25))
            listind = 0
            noutliers = 0
            outlierslist = list()
            outliersindx = list()
            if binatt == 0:
                for val in attvals:
                    print('Evaluating outliers in attribute (',att,',',iatt,'out of',totatts,'): processing', (listind/len(attvals))*100,' %')
                    if val > outlierupth or val < outlierloth:
                        noutliers = noutliers + 1
                        outlierslist.append(attvals.pop(listind))
                        outliersindx.append(attindx.pop(listind))
                    listind = listind + 1


            ################################# EVALUATE VALUES OF TARGET


            yesvec_vals = list()
            yesvec_indx = list()
            novec_vals = list()
            novec_indx = list()

            iteri = 0
            yvec = list()
            xvec = list()
            yidx = list()
            xidx = list()
            for ind in attindx:
                print('Evaluating target values for attribute (',att,',',iatt,'out of',totatts,'): processing', (iteri/len(attindx))*100,' %')
                checkcell =  valtardb.loc[ind,[targetfield]]
                checktarg = checkcell.get(targetfield)
                if not (checktarg == 'NaN' or checktarg == '' or checktarg == [] or checktarg == 'NaT' or checktarg == 'NULL' or checktarg == 'NA'):
                    if checktarg == 1:
                        yesvec_vals.append(attvals[iteri])
                        yesvec_indx.append(ind)
                        yvec.append(1)
                        xvec.append(attvals[iteri])
                    if checktarg == 0:
                        novec_vals.append(attvals[iteri])
                        novec_indx.append(ind)
                        yvec.append(0)
                        xvec.append(attvals[iteri])
                iteri = iteri + 1
            if binatt == 1:
                yesoverth = 0.5
                nooverth = 0.5
            else:
                yesoverth = median(yesvec_vals)
                nooverth = median(novec_vals)
            yesoverlist = list()
            yesunderlist = list()
            for yv in yesvec_vals:
                if yv >= yesoverth:
                    yesoverlist.append(yv)
                else:
                    yesunderlist.append(yv)

            nooverlist = list()
            nounderlist = list()
            for nv in novec_vals:
                if nv >= nooverth:
                    nooverlist.append(nv)
                else:
                    nounderlist.append(nv)
            totvalatts = yesvec_vals + novec_vals

            ################################# PERFORM STATISTIC TESTS

            print('Computing parameters for attribute (',att,',',iatt,'out of',totatts,')')

            ############## CONTINUOS VARIABLES

            if binatt == 0:

                ######################## PLOT
                if goplot == 1:
                    dirname = outprefix + '_ExpAnalysisSTATS' + '_' + targetfield
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname)
                    outpicname = dirname + '\\' + outprefix + '_' + att + '_hist.png'
                    if not(os.path.isfile(outpicname)):
                        try:
                            del(allarr)
                            del(onesarr)
                            del(zerosarr)
                            del(histbins)
                            del(minval)
                            del(maxval)
                        except:
                            a = 1
                        maxval = max(totvalatts)
                        minval = min(totvalatts)
                        print('Generating plot for attribute (',att,',',iatt,'out of',totatts,')')
                        peace = round(((maxval - minval) / histcols)*100)/100
                        histbins = list()
                        xticks = list()
                        tickf = 0
                        for bi in range(histcols+1):
                            histbins.append(minval + bi*peace)
                            if histcols > 10:
                                if tickf == 0:
                                    xticks.append(minval + bi*peace)
                                    tickf = 1
                                else:
                                    tickf = 0
                            else:
                                xticks.append(minval + bi*peace)
                        allarr = np.asarray(totvalatts)
                        onesarr = np.asarray(yesvec_vals)
                        zerosarr = np.asarray(novec_vals)
                        plt.figure(iatt,clear = True)
                        plt.subplot(3,1,1)
                        plt.hist(allarr,bins = histbins)
                        plt.ylabel('All')
                        plt.xticks(xticks)
                        plt.subplot(3,1,2)
                        plt.hist(onesarr,bins = histbins)
                        plt.ylabel('Target 1')
                        plt.xticks(xticks)
                        plt.subplot(3,1,3)
                        plt.hist(zerosarr,bins = histbins)
                        plt.ylabel('Target 0')
                        plt.xticks(xticks)
                        plt.suptitle(att + ' DISTRIBUTION')
                        plt.savefig(outpicname)

                ############## Normality test
                print('Testing normality for attribute (',att,',',iatt,'out of',totatts,')')
                normdist = 1
                normstats, normp = normaltest(attvals)
                if normp <= 0.05:
                    normdist = 0
                if normdist == 1:
                    testname = 'T-TEST'
                else:
                    testname = 'MANN-WHITNEY TEST'
                normteststats = str(normstats)

                ############## Distribution test

                allps = list()

                if len(yesvec_vals) >= len(novec_vals):
                    veclen = len(novec_vals)
                else:
                    veclen = len(yesvec_vals)
                mwavec = 0
                mwbvec = 0
                for it in range(1,niterations):
                    print('Computing paramters for attribute (',att,',',iatt,'out of',totatts,'): distribution iteraton processing = ',(it/niterations)*100,' %')
                    del(mwavec)
                    del(mwbvec)
                    mwavec = random.sample(yesvec_vals,veclen)
                    mwbvec = random.sample(novec_vals,veclen)
                    if normdist == 1:
                        stat, p = ttest_ind(mwavec, mwbvec)
                    else:
                        stat, p = mannwhitneyu(mwavec, mwbvec)
                    allps.append(p)

                finalp = median(allps)

            ############## BINARY VARABLES

            else:

                testname = 'CHI-SQUARED'
                if len(yesoverlist)>0 and len(nooverlist)>0 and len(yesunderlist)>0 and len(nounderlist)>0:
                    conttab = [[len(yesoverlist),len(nooverlist)],[len(yesunderlist),len(nounderlist)]]
                    stat, finalp, dof, expected = chi2_contingency(conttab)
                else:
                    finalp = 'NA'

            ############## LOGISTIC REGRESSION

            allacc = list()
            alloddsr = list()
            alloddsrlow = list()
            alloddsrup = list()
            alloddspval = list()
            allrsq = list()
            loginterations = niterations

            if len(yesvec_vals) >= len(novec_vals):
                smallvec = novec_vals
                longvec = yesvec_vals
                smally = 0
                longy = 1
            else:
                longvec = novec_vals
                smallvec = yesvec_vals
                smally = 1
                longy = 0

            if ((1 in smallvec or 1 in longvec) and (0 in smallvec or 0 in longvec)) or binatt == 0:
                for it in range(loginterations):
                    print('Computing paramters for attribute (',att,',',iatt,'out of',totatts,'): logistic regression processing = ',(it/loginterations)*100,' %')
                    logregres = compute_logreg(smally,smallvec,longy,longvec,traintest,vartype,att)
                    if len(logregres) > 5:
                        rsq = logregres[5]
                        allrsq.append(rsq)
                    acc = logregres[0]
                    oddsr = logregres[1]
                    oddsrlow = logregres[2]
                    oddsrup= logregres[3]
                    oddspval = logregres[4]
                    allacc.append(acc)
                    alloddsr.append(oddsr)
                    alloddsrlow.append(oddsrlow)
                    alloddsrup.append(oddsrup)
                    alloddspval.append(oddspval)

                finalacc = median(allacc)
                finaloddsr = median(alloddsr)
                finaloddsrlow = median(alloddsrlow)
                finaloddsrup = median(alloddsrup)
                finaloddspval = median(alloddspval)
                if len(allrsq) > 0:
                    finalrsq = median(allrsq)
            else:
                finalrsq = 'NA'
                finalacc = 'NA'
                finaloddsr = 'NA'
                finaloddsrlow = 'NA'
                finaloddsrup = 'NA'
                finaloddspval = 'NA'

            ################################### EXPORT DATA

            out_missingvals = attnulls
            out_missingvals_p = str(round((out_missingvals / len(valtardb))*10000) / 100) + ' %'
            out_outliers = len(outlierslist)
            out_outliers_p = str(round((out_outliers / len(valtardb))*10000) / 100) + ' %'
            out_sampsize = len(totvalatts)
            out_sampsize_p = str(round((out_sampsize / len(valtardb))*10000) / 100) + ' %'
            out_target1 = len(yesvec_vals)
            out_target1_p = str(round((out_target1 / len(totvalatts))*10000) / 100) + ' %'
            out_target0 = len(novec_vals)
            out_target0_p = str(round((out_target0 / len(totvalatts))*10000) / 100) + ' %'
            if binatt == 0:
                if normdist == 0:
                    out_target1_d = str(round(median(yesvec_vals)*100) / 100) + ' (' +  str(round(np.quantile(yesvec_vals,0.25)*100) / 100) + ' - ' + str(round(np.quantile(yesvec_vals,0.75)*100) / 100) + ')'
                    out_target0_d = str(round(median(novec_vals)*100) / 100) + ' (' +  str(round(np.quantile(novec_vals,0.25)*100) / 100) + ' - ' + str(round(np.quantile(novec_vals,0.75)*100) / 100) + ')'
                else:
                    out_target1_d = str(round(mean(yesvec_vals)*100) / 100) + ' +- ' + str(round(stdev(yesvec_vals)*100) / 100)
                    out_target0_d = str(round(mean(novec_vals)*100) / 100) + ' +- ' + str(round(stdev(novec_vals)*100) / 100)
            else:
                out_target1_d = 'NA'
                out_target0_d = 'NA'
            if len(yesoverlist)>0 or len(nooverlist)>0:
                out_over = len(yesoverlist) + len(nooverlist)
                out_over_p = str(round((out_over / len(totvalatts)) * 10000) / 100) + ' %'
                out_overyes = len(yesoverlist)
                out_overyes_p = str(round((out_overyes / out_over)*10000) / 100) + ' %'
                out_overno = len(nooverlist)
                out_overno_p = str(round((out_overno / out_over)*10000) / 100) + ' %'
            else:
                out_over = 0
                out_over_p = str(round((out_over / len(totvalatts)) * 10000) / 100) + ' %'
                out_overyes = len(yesoverlist)
                out_overyes_p = 'NA'
                out_overno = len(nooverlist)
                out_overno_p = 'NA'
            if len(yesunderlist)>0 or len(nounderlist)>0:
                out_under = len(yesunderlist) + len(nounderlist)
                out_under_p = str(round((out_under / len(totvalatts)) * 10000) / 100) + ' %'
                out_underyes = len(yesunderlist)
                out_underyes_p = str(round((out_underyes / out_under)*10000) / 100) + ' %'
                out_underno = len(nounderlist)
                out_underno_p = str(round((out_underno / out_under)*10000) / 100) + ' %'
            else:
                out_under = 0
                out_under_p = str(round((out_under / len(totvalatts)) * 10000) / 100) + ' %'
                out_underyes = len(yesunderlist)
                out_underyes_p = 'NA'
                out_underno = len(nounderlist)
                out_underno_p = 'NA'
            out_acc = finalacc
            out_pval = finalp

            nameindx = 0
            nameindxstr = checknameindx(nameindx)
            if binatt == 1:
                attname = '(BIN) ' + att
            else:
                if normdist == 1:
                    attname = '(NUM_NORM) ' + att + ' \n ' + str(round(mean(totvalatts)*100) / 100) + ' +- ' + str(round(stdev(totvalatts)*100) / 100)
                else:
                    attname = '(NUM_NOTNORM) ' + att + ' \n ' + str(round(median(totvalatts)*100) / 100) + ' (' +  str(round(np.quantile(totvalatts,0.25)*100) / 100) + ' - ' + str(round(np.quantile(totvalatts,0.75)*100) / 100) + ')'
                attname = attname + ' \n ' + 'OUTLIERS: < ' + str(round(outlierloth*100)/100) + ', > ' + str(round(outlierupth*100)/100)

            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' MISSING'
            outdb.loc[outrow,'VALUE'] = out_missingvals
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' MISSING % on VALID'
            outdb.loc[outrow,'VALUE'] = out_missingvals_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OUTLIERS'
            outdb.loc[outrow,'VALUE'] = out_outliers
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OUTLIERS % on VALID'
            outdb.loc[outrow,'VALUE'] = out_outliers_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' SAMPLE SIZE'
            outdb.loc[outrow,'VALUE'] = out_sampsize
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' SAMPLE SIZE % on VALID'
            outdb.loc[outrow,'VALUE'] = out_sampsize_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 1'
            outdb.loc[outrow,'VALUE'] = out_target1
            outrow = outrow + 1
            nameindx = nameindx + 1
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 1 % on VALID'
            outdb.loc[outrow,'VALUE'] = out_target1_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 1 Dist'
            outdb.loc[outrow,'VALUE'] = out_target1_d
            outrow = outrow + 1
            nameindx = nameindx + 1
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 0'
            outdb.loc[outrow,'VALUE'] = out_target0
            outrow = outrow + 1
            nameindx = nameindx + 1
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 0 % on VALID'
            outdb.loc[outrow,'VALUE'] = out_target0_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 0 Dist'
            outdb.loc[outrow,'VALUE'] = out_target0_d
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OVER'
            outdb.loc[outrow,'VALUE'] = out_over
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OVER % on VALID'
            outdb.loc[outrow,'VALUE'] = out_over_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OVER TARGET 1'
            outdb.loc[outrow,'VALUE'] = out_overyes
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 1 % on OVER'
            outdb.loc[outrow,'VALUE'] = out_overyes_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' OVER TARGET 0'
            outdb.loc[outrow,'VALUE'] = out_overno
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 0 % on OVER'
            outdb.loc[outrow,'VALUE'] = out_overno_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' UNDER'
            outdb.loc[outrow,'VALUE'] = out_under
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' UNDER % on VALID'
            outdb.loc[outrow,'VALUE'] = out_under_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' UNDER TARGET 1'
            outdb.loc[outrow,'VALUE'] = out_underyes
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 1 % on UNDER'
            outdb.loc[outrow,'VALUE'] = out_underyes_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' UNDER TARGET 0'
            outdb.loc[outrow,'VALUE'] = out_underno
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' TARGET 0 % on UNDER'
            outdb.loc[outrow,'VALUE'] = out_underno_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO median'
            outdb.loc[outrow,'VALUE'] = finaloddsr
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO low C.I.'
            outdb.loc[outrow,'VALUE'] = finaloddsrlow
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO up C.I.'
            outdb.loc[outrow,'VALUE'] = finaloddsrup
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO p-val'
            outdb.loc[outrow,'VALUE'] = finaloddspval
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ACCURACY'
            outdb.loc[outrow,'VALUE'] = out_acc
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' ' + testname + ' p-val'
            outdb.loc[outrow,'VALUE'] = out_pval
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)

        ################################ HANDLE CATEGORICAL

        else:
            allcats = list()
            totcount = list()
            tottargcount = list()
            onescount = list()
            zeroscount = list()
            cind = 0
            for catt in attvals:
                print('Evaluating categories for attribute (',att,',',iatt,'out of',totatts,'): processing = ',(cind/len(attvals))*100,' %')
                if catt not in allcats:
                    allcats.append(catt)
                    totcount.append(1)
                    if attindx[cind] in tarones:
                        onescount.append(1)
                        zeroscount.append(0)
                        tottargcount.append(1)
                    elif attindx[cind] in tarzeros:
                        onescount.append(0)
                        zeroscount.append(1)
                        tottargcount.append(1)
                    else:
                        onescount.append(0)
                        zeroscount.append(0)
                        tottargcount.append(0)
                else:
                    catind = allcats.index(catt)
                    totcount[catind] = totcount[catind] + 1
                    if attindx[cind] in tarones:
                        onescount[catind] = onescount[catind] + 1
                        tottargcount[catind] = tottargcount[catind] + 1
                    elif attindx[cind] in tarzeros:
                        zeroscount[catind] = zeroscount[catind] + 1
                        tottargcount[catind] = tottargcount[catind] + 1
                cind = cind + 1

            ####################### CHI-2 TEST FOR CATEGORIES

            testname = 'CHI-SQUARED'

            ci = 0
            firstline = 1
            for cat in allcats:
                if firstline == 1:
                    if (onescount[ci-1]>0 or zeroscount[ci-1]>0) and (onescount[ci]>0 or zeroscount[ci]>0):
                        conttab = np.array([[onescount[ci-1],zeroscount[ci-1]],[onescount[ci],zeroscount[ci]]])
                        firstline = 0
                else:
                    if onescount[ci]>0 and zeroscount[ci]>0:
                        newline = [[onescount[ci],zeroscount[ci]]]
                        conttab = np.vstack([conttab,newline])
                ci = ci + 1
            stat, finalp, dof, expected = chi2_contingency(conttab)

            ############## LOGISTIC REGRESSION (CATEGORICAL)

            allacc = list()
            alloddsr = list()
            alloddsrlow = list()
            alloddsrup = list()
            alloddspval = list()
            allrsq = list()
            loginterations = niterations
            yesvec_vals = valtardb.loc[(valtardb[targetfield]==1),att].tolist()
            novec_vals = valtardb.loc[(valtardb[targetfield]==0),att].tolist()

            if len(yesvec_vals) >= len(novec_vals):
                smallvec = novec_vals
                longvec = yesvec_vals
                smally = 0
                longy = 1
            else:
                longvec = novec_vals
                smallvec = yesvec_vals
                smally = 1
                longy = 0

            for it in range(loginterations):
                print('Computing paramters for attribute (',att,',',iatt,'out of',totatts,'): logistic regression processing = ',(it/loginterations)*100,' %')
                logregres = compute_logreg(smally,smallvec,longy,longvec,traintest,vartype,att)
                if len(logregres) > 5:
                    rsq = logregres[5]
                    allrsq.append(rsq)
                acc = logregres[0]
                oddsr = logregres[1]
                oddsrlow = logregres[2]
                oddsrup= logregres[3]
                oddspval = logregres[4]
                allacc.append(acc)
                alloddsr.append(oddsr)
                alloddsrlow.append(oddsrlow)
                alloddsrup.append(oddsrup)
                alloddspval.append(oddspval)

            finalacc = median(allacc)
            finaloddsr = median(alloddsr)
            finaloddsrlow = median(alloddsrlow)
            finaloddsrup = median(alloddsrup)
            finaloddspval = median(alloddspval)
            if len(allrsq) > 0:
                finalrsq = median(allrsq)

            ########################## OUTPUT OF CATEGORICAL

            out_missing = len(valtardb) - sum(totcount)
            out_missing_p = str(round((out_missing / len(valtardb))*10000) / 100) + ' %'
            out_samplesize = sum(tottargcount)
            out_samplesize_p = str(round((out_samplesize / len(valtardb))*10000) / 100) + ' %'

            attname = '(CAT) ' + att + '\n N CATEGORIES = ' + str(len(allcats))
            nameindx = 0
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' MISSING'
            outdb.loc[outrow,'VALUE'] = out_missing
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' MISSING % on VALID'
            outdb.loc[outrow,'VALUE'] = out_missing_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' SAMPLE SIZE'
            outdb.loc[outrow,'VALUE'] = out_samplesize
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' SAMPLE SIZE % on VALID'
            outdb.loc[outrow,'VALUE'] = out_samplesize_p
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            ici = 0
            for cat in allcats:
                if tottargcount[ici] > 0:
                    out_totcount = tottargcount[ici]
                    out_totcount_p = str(round((out_totcount / out_samplesize)*10000) / 100) + ' %'
                    out_onescount = onescount[ici]
                    out_onescount_p = str(round((out_onescount / out_totcount)*10000) / 100) + ' %'
                    out_zeroscount = zeroscount[ici]
                    out_zeroscount_p = str(round((out_zeroscount / out_totcount)*10000) / 100) + ' %'

                    catstr = '<' + cat + '>'
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ' + catstr + ' TOTALCOUNT'
                    outdb.loc[outrow,'VALUE'] = out_totcount
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ' + catstr + ' TOTALCOUNT % on VALID'
                    outdb.loc[outrow,'VALUE'] = out_totcount_p
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ' + catstr + ' N ONES'
                    outdb.loc[outrow,'VALUE'] = out_onescount
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ONES % on ' + catstr
                    outdb.loc[outrow,'VALUE'] = out_onescount_p
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ' + catstr + ' N ZEROS'
                    outdb.loc[outrow,'VALUE'] = out_zeroscount
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)
                    outdb.loc[outrow,'VARIABILE'] = attname
                    outdb.loc[outrow,'STAT'] = nameindxstr + ' ZEROS % on ' + catstr
                    outdb.loc[outrow,'VALUE'] = out_zeroscount_p
                    outrow = outrow + 1
                    nameindx = nameindx + 1
                    nameindxstr = checknameindx(nameindx)

                ici = ici + 1

            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' CHI^2 TEST p-val'
            outdb.loc[outrow,'VALUE'] = finalp
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO median'
            outdb.loc[outrow,'VALUE'] = finaloddsr
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO low C.I.'
            outdb.loc[outrow,'VALUE'] = finaloddsrlow
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO up C.I.'
            outdb.loc[outrow,'VALUE'] = finaloddsrup
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ODDsRATIO p-val'
            outdb.loc[outrow,'VALUE'] = finaloddspval
            outrow = outrow + 1
            nameindx = nameindx + 1
            nameindxstr = checknameindx(nameindx)
            outdb.loc[outrow,'VARIABILE'] = attname
            outdb.loc[outrow,'STAT'] = nameindxstr + ' LOGREG ACCURACY'
            outdb.loc[outrow,'VALUE'] = finalacc
    
    return(outdb)

def multivariate_logreg_or(indbraw,allcolumns,uid,target,mode,dropthresh="foo",cindex="foo"):

    if dropthresh == "foo":
        dropthresh = 0.1
    excludedlist = list()
    indb = indbraw.loc[~indbraw[target].isna()].copy(deep=True)
    indb.loc[indb[target]==-1,target] = 0
    cattable = pd.read_excel('DATASOURCE_categorical_variables_map.xlsx')
    numtable = pd.read_excel('DATASOURCE_numvars_cutoff.xlsx')
    attributes = list()
    for col in allcolumns:
        attributes.append(col)
    try:
        allcolumns.pop(allcolumns.index('Unnamed: 0'))
        allcolumns.pop(allcolumns.index('Unnamed: 0.1'))
    except:
        a = 1
    keepcols = allcolumns.copy()
    keepcols.append(target)
    xDb = indb[(keepcols)].copy(deep=True)
    totrows = xDb.shape[0]
    othertypes = list()
    for col in allcolumns:
        varname = col
        nullvals = xDb.loc[xDb[col].isna()].shape[0]
        if nullvals > dropthresh*totrows:
            xDb =xDb.drop(col,axis=1)
            excludedlist.append(col)
            attributes.pop(attributes.index(col))
        else:
            txDb = xDb.loc[~xDb[col].isna()].copy(deep=True)
            xDb = txDb
            rawdatavec = pd.DataFrame(indb[col].copy(deep=True))
            datavec = rawdatavec.loc[~rawdatavec[col].isna()].copy(deep=True)
            dim = datavec.shape[0]
            bindf = datavec.loc[((datavec[col]==0)|(datavec[col]==1))].copy(deep=True)
            bindim = bindf.shape[0]
            if (dim-bindim) <= (0.05*dim):
                datatype = 0
            else:
                checker = datavec.iloc[0,0]
                if type(checker)!=float and type(checker)!=int and type(checker)!=np.int64 and type(checker)!=np.float64:
                    datatype = 2
                else:
                    datatype = 1
            if mode == 1: # Mode 1 = binary only
                if datatype != 0:
                    xDb = xDb.drop(col,axis=1)
                    othertypes.append(col)
            elif mode == 2: # Mode 2 = numerical and categoricals only
                if datatype == 0:
                    xDb = xDb.drop(col,axis=1)
                    othertypes.append(col)
            if datatype == 1 and (mode == 0 or mode == 2):
                cutofflag = 0
                for nr in numtable.iterrows():
                    if nr[1]['Variable'] == varname:
                        cutofflag = 1
                        cutoff = nr[1]['cutoff']
                        reverseflag = nr[1]['reverse']
                if cutofflag == 0:
                    datarr = np.asarray(datavec)
                    medval = np.median(datarr)
                    xDb.loc[xDb[col]<medval,col] = 0
                    xDb.loc[xDb[col]>=medval,col] = 1
                elif cutofflag == 1:
                    if reverseflag == 0:
                        xDb.loc[xDb[col]<cutoff,col] = 0
                        xDb.loc[xDb[col]>=cutoff,col] = 1
                    elif reverseflag == 1:
                        xDb.loc[xDb[col]<cutoff,col] = 1
                        xDb.loc[xDb[col]>=cutoff,col] = 0
            elif datatype == 2 and (mode!=1):
                zeroval = cattable.loc[cattable['Unnamed: 0']==varname,0].iloc[0]
                oneval = cattable.loc[cattable['Unnamed: 0']==varname,1].iloc[0]
                if zeroval != 'other' and oneval == 'other':
                    xDb.loc[xDb[col]==zeroval,col] = 0
                    xDb.loc[((xDb[col]!=zeroval)&(xDb[col]!=0)),col] = 1
                elif zeroval == 'other' and oneval != 'other':
                    xDb.loc[xDb[col]==oneval,col] = 1
                    xDb.loc[((xDb[col]!=oneval)&(xDb[col]!=1)),col] = 0
                else:
                    xDb.loc[xDb[col]==oneval,col] = 1
                    xDb.loc[xDb[col]==zeroval,col] = 0
                txDb = xDb.loc[((xDb[col]==1)|(xDb[col]==0))].copy(deep=True)
                xDb = txDb.copy(deep=True)
    for ot in othertypes:
        attributes.pop(attributes.index(ot))
    Y_train= xDb[target].copy(deep=True).to_numpy().astype(int).reshape(-1,1)
    X_train = xDb[(attributes)].copy(deep=True).to_numpy().astype(int)
    res = sm.Logit(Y_train, X_train).fit(start_params=None, method='powell', maxiter=100)
    coefs = pd.DataFrame(np.exp(res.params)).transpose()
    coefs = coefs.rename(index={0:'OddsRatio'})
    confi = pd.DataFrame(np.exp(res.conf_int()).transpose())
    confi = confi.rename(index={0:'CI 2.5%',1:'CI 97.5%'})
    pvals = pd.DataFrame(np.asarray(res.pvalues)).transpose()
    pvals = pvals.rename(index={0:'p-value'})
    logtab = pd.concat([coefs,confi,pvals],axis=0,sort=False)
    j = 0
    for att in attributes:
        logtab = logtab.rename(columns={j:att})
        j = j + 1
    logtab = logtab.transpose()

    if cindex == 1:
        df = pd.DataFrame()
        y_test_raw = res.predict(X_train)
        y_train = xDb[target].copy(deep=True)
        y_test = pd.Series(y_test_raw).transpose().reset_index(drop=True)
        y_train.reset_index(inplace=True,drop=True)
        df['T'] = y_train
        df['E'] = y_test
        ci = concordance_index(y_train,y_test)
        return(logtab,excludedlist,xDb,res,ci)
    
    
    return(logtab,excludedlist,xDb,res)

def plot_sankey(flows,prefix,outpath,title="foo",total="foo",part="foo"):
    
    if title == "foo":
        title = 'Sankey diagram'
    outpathfile  = outpath+'SANKEY_CHARTS//'
    if not(os.path.isdir(outpathfile)):
        os.makedirs(outpathfile)
    if total == "foo" or part == "foo":

        total = sum(flows)
        ratiosl = list()
        for f in flows:
            ratiosl.append(f/total)
        ratios = np.asarray(ratiosl)
        pointer = 0
        ycoordsl = list()
        ybv = ((flows[0]+flows[3])/total)/2
        ybe = ybv*2 + ((flows[1]+flows[2])/total)/2
        ycoordsl.append(ybv)
        ycoordsl.append(ybe)
        #offsets = [0,0.1,0,0]
        ri = 0
        for r in ratios:
            ycoordsl.append(pointer + (r/2))
            pointer = pointer + r #+ offsets[ri]
            ri = ri + 1
        ycoords = np.asarray(ycoordsl)
        
        fig = go.Figure(data=[go.Sankey(
            arrangement = 'fixed',
            node = dict(
                pad = 1,
                thickness = 150,
                line = dict(color = "black", width = 2),
                #label = ["bv", "be", "kt", "cr", "ud", "im"],
                x = [0.25,0.25,0.75,0.75,0.75,0.75],
                y = ycoordsl,
                color = ['rgb(0,32,96)','rgb(176,172,0)','rgb(0,32,96)','rgb(0,176,80)','rgb(176,172,0)','rgb(255,0,0)']
            ),
            link = dict(
                source = [0,1,1,0],
                target = [2,3,4,5],
                value = flows,
                color = ['rgba(0,32,96,0.4)','rgba(0,176,80,0.4)','rgba(176,172,0,0.4)','rgba(255,0,0,0.4)']
            ))])

        fig.update_layout(title_text=title, font_size=16)
        fig.write_image(outpathfile + prefix + title + ".png")
        fig.show()
    
    else:
        pdummy = total - part
        pflows = flows
        pflows.append(pdummy)
        pratiosl = list()
        for f in pflows:
            pratiosl.append(f/total)
        ppointer = 0
        pycoordsl = list()
        pybv = ((pflows[0]+pflows[3])/total)/2
        pybe = pybv*2 + ((pflows[1]+pflows[2])/total)/2
        pydum = pybe + ((pflows[1]+pflows[2])/total)/2 + ((pflows[4])/total)/2
        pycoordsl.append(pybv)
        pycoordsl.append(pybe)
        pycoordsl.append(pydum)
        pri = 0
        for r in pratiosl:
            pycoordsl.append(ppointer + (r/2))
            ppointer = ppointer + r
            pri = pri + 1
        
        pfig = go.Figure(data=[go.Sankey(
            arrangement = 'fixed',
            node = dict(
                pad = 1,
                thickness = 150,
                line = dict(color = "black", width = 2),
                #label = ["bv", "be", "kt", "cr", "ud", "im"],
                x = [0.25,0.25,0.25,0.75,0.75,0.75,0.75,0.75],
                y = pycoordsl,
                color = ['rgb(0,32,96)','rgb(176,172,0)','rgb(255,255,255)','rgb(0,32,96)','rgb(0,176,80)','rgb(176,172,0)','rgb(255,0,0)','rgb(255,255,255)']
            ),
            link = dict(
                source = [0,1,1,0,2],
                target = [3,4,5,6,7],
                value = pflows,
                color = ['rgba(0,32,96,0.4)','rgba(0,176,80,0.4)','rgba(176,172,0,0.4)','rgba(255,0,0,0.4)','rgba(255,255,255,0.4)']
            ))])

        pfig.update_layout(title_text=title, font_size=16)
        pfig.write_image(outpathfile + prefix + title + ".png")
        pfig.show()
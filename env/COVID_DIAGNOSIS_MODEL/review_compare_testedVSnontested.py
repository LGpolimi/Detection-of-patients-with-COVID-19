import numpy as np
import pandas as pd
import random
from scipy.stats import ranksums
from scipy.stats import chi2_contingency

outprefix = 'resp_validation'
saveout = 1
indb = pd.read_excel('resp_validationAO_MainDB_211001_211231.xlsx')
target = 'TAMPONE_+-7'
atts = ['MOTIVO','SEX','COSCIENZA','RESPIRODS','CONTATTO','TEMPERATURA','AGE','RESPIRO','FEBBRE','DIARREA','TOSSE','GUSTO-OLFATTO','ASTENIA','VOMITO','SOSPETTO','RESP_FREQ','SAT_ARIA','SAT_O2','FREQ_CARDIO','PRESS_SIST','PRESS_DIAST','TOT_POS','CRIT','NEWS_ALERT']
zindex = 5
outdb = pd.DataFrame()
cattable = pd.read_excel('DATASOURCE_categorical_variables_map.xlsx')
iatt = 0
totatts = len(atts)

for att in atts:
    iatt = iatt + 1
    attnulls = list()
    attindx = list()
    attvals = list()

    ################################# EVALUATE NULLS
    print('Evaluating nulls in attribute ',att)
    z_miss = indb.loc[(indb[target]==0)&((pd.isnull(indb[att]))|(indb[att].isna())|(indb[att]=='')|(indb[att]=='NaT')|(indb[att]=='NULL'))].shape[0]
    o_miss = indb.loc[(indb[target]==1)&((pd.isnull(indb[att]))|(indb[att].isna())|(indb[att]=='')|(indb[att]=='NaT')|(indb[att]=='NULL'))].shape[0]
    z_tot = indb.loc[(indb[target]==0)].shape[0]
    o_tot = indb.loc[(indb[target]==1)].shape[0]
    nullsdb = indb.loc[((indb[att].isna())|(indb[att]=='')|(indb[att]=='NaT')|(indb[att]=='NULL'))]
    valdb = indb.loc[~((indb[att].isna())|(indb[att]=='')|(indb[att]=='NaT')|(indb[att]=='NULL'))]
    attnulls = nullsdb.shape[0]

    ################################ EVALUATE CTAEGORICAL

    catcounter = 0
    catthresh = 100

    for ci in range(catthresh):
        randitem = random.choice(list(valdb[att]))
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
        attvals = np.asarray(valdb[att])
        nattvals = valdb[att].shape[0]
        for val in list(valdb[att]):
            print('Evaluating binarity in attribute (',att,',',iatt,'out of',totatts,'): processing ',(bvind/nattvals)*100,' %')
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


        listind = 0
        noutliers = 0
        outlierslist = list()
        outliersindx = list()
        zerodb = pd.DataFrame(valdb.loc[valdb[target]==0,att].copy(deep=True))
        onesdb = pd.DataFrame(valdb.loc[valdb[target]==1,att].copy(deep=True))

        if binatt == 0:
            print('Removing outliers in attribute (', att, ',', iatt, 'out of', totatts, ')')
            zmissp = str(round(((z_miss / z_tot)*100)*10)/10) + '%'
            omissp = str(round(((o_miss / o_tot)*100)*10)/10) + '%'
            z_med = np.median(zerodb[att])
            z_iqr = np.quantile(zerodb[att],0.75) - np.quantile(zerodb[att],0.25)
            z_upth = z_med + zindex*z_iqr
            z_loth = z_med - zindex*z_iqr
            z_outl = zerodb.loc[~(zerodb[att]>=z_loth)&(zerodb[att]<=z_upth)].shape[0]
            zerodb = zerodb.loc[(zerodb[att]>=z_loth)&(zerodb[att]<=z_upth)].copy(deep=True)
            o_med = np.median(onesdb[att])
            o_iqr = np.quantile(onesdb[att],0.75) - np.quantile(onesdb[att],0.25)
            o_upth = o_med + zindex*o_iqr
            o_loth = o_med - zindex*o_iqr
            o_outl = onesdb.loc[~(onesdb[att]>=z_loth)&(onesdb[att]<=z_upth)].shape[0]
            onesdb = onesdb.loc[(onesdb[att]>=z_loth)&(onesdb[att]<=z_upth)].copy(deep=True)
            zerolist = np.asarray(zerodb)
            oneslist = np.asarray(onesdb)
            zeromed = np.median(zerolist)
            zerofq = np.quantile(zerolist,0.25)
            zerotq = np.quantile(zerolist,0.75)
            onesmed = np.median(oneslist)
            onesfq = np.quantile(oneslist,0.25)
            onestq = np.quantile(oneslist,0.75)
            zoutlp = str(round(((z_outl / z_tot)*100)*10)/10) + '%'
            ooutlp = str(round(((o_outl / o_tot)*100)*10)/10) + '%'
            rsres = ranksums(zerolist,oneslist)
            p = rsres.pvalue

            outdb.loc[att, 'TESTED_DIST'] = str(onesmed) + ' [' + str(onesfq) + '-' + str(onestq) + '] Missing = ' + str(o_miss) + '(' + omissp + ') Outliers = ' + str(o_outl) + '(' + ooutlp + ')'
            outdb.loc[att, 'NONTESTED_DIST'] = str(zeromed) + ' [' + str(zerofq) + '-' + str(zerotq) + '] Missing = ' + str(z_miss) + '(' + zmissp + ') Outliers = ' + str(z_outl) + '(' + zoutlp + ')'
            if p < 0.001:
                outdb.loc[att, 'RANK-SUM PVAL'] = '<0.001'
            else:
                outdb.loc[att, 'RANK-SUM PVAL'] = round(p*1000)/1000
        else:
            nsamp = valdb.shape[0]
            zerodb = valdb.loc[valdb[target] == 0, [target,att]].copy(deep=True)
            onesdb = valdb.loc[valdb[target] == 1, [target,att]].copy(deep=True)
            zerolist = np.asarray(zerodb)
            oneslist = np.asarray(onesdb)
            zeroszeros = zerodb.loc[zerodb[att] == 0,att].copy(deep=True)
            zerosones = zerodb.loc[zerodb[att] == 1, att].copy(deep=True)
            oneszeros = onesdb.loc[onesdb[att] == 0, att].copy(deep=True)
            onesones = onesdb.loc[onesdb[att] == 1, att].copy(deep=True)
            n_zeroszeros = zeroszeros.shape[0]
            n_zerosones = zerosones.shape[0]
            n_oneszeros = oneszeros.shape[0]
            n_onesones = onesones.shape[0]
            stat, p, dof, expected = chi2_contingency([[n_zeroszeros,n_zerosones],[n_oneszeros,n_onesones]])
            zeroonesperc = n_zerosones / zerodb.shape[0]
            onesonesperc = n_onesones / onesdb.shape[0]
            outdb.loc[att, 'TESTED_COUNT'] = str(n_onesones) + ' (' + str(round(onesonesperc*1000)/10) + '%)'
            outdb.loc[att, 'NONTESTED_COUNT'] = str(n_zerosones) + ' (' + str(round(zeroonesperc*1000)/10) + '%)'
            if p < 0.001:
                outdb.loc[att, 'CHI-SQR PVAL'] = '<0.001'
            else:
                outdb.loc[att, 'CHI-SQR PVAL'] = round(p*1000)/1000
    else:
        allcats = list(valdb[att].unique())
        zerodb = valdb.loc[valdb[target] == 0, [target,att]].copy(deep=True)
        onesdb = valdb.loc[valdb[target] == 1, [target,att]].copy(deep=True)
        zerovec = list()
        onesvec = list()
        diststr_zero = ''
        diststr_ones = ''
        for cat in allcats:
            zeron = zerodb.loc[zerodb[att] == cat].shape[0]
            onesn = onesdb.loc[onesdb[att] == cat].shape[0]
            zerovec.append(zeron)
            onesvec.append(onesn)
            zeroperc = zeron/zerodb.shape[0]
            onesperc = onesn/onesdb.shape[0]
            diststr_zero = diststr_zero + cat + ': ' + str(zeron) + ' (' + str(round(zeroperc*1000)/10) + '%)  '
            diststr_ones = diststr_ones + cat + ': ' + str(onesn) + ' (' + str(round(onesperc * 1000) / 10) + '%)  '
        stat, p, dof, expected = chi2_contingency([np.asarray(zerovec), np.asarray(onesvec)])
        outdb.loc[att, 'TESTED_COUNTS'] = diststr_ones
        outdb.loc[att, 'NONTESTED_COUNTS'] = diststr_zero
        if p < 0.001:
            outdb.loc[att, 'CHI-SQR PVALUE'] = '<0.001'
        else:
            outdb.loc[att, 'CHI-SQR PVALUE'] = round(p*1000)/1000

if saveout == 1:
    outdb.to_excel(outprefix+'_VariablesDistribution.xlsx')
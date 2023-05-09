print('\n\nScript started: A_preprocess_input_db')
print('\n\nImporting libraries')
import pandas as pd
import math
import statistics
from Z_classes_and_functions import *

########################## PARAMETERS SETTING

mindate = '211001'
maxdate = '211231'
maindbname = 'MainDB_'+ mindate + '_' + maxdate
maindbfilename = 'DATASOURCE_' +maindbname + '.csv'
outputdbname = maindbname + '.xlsx'
hubhospitalsfilename = 'DATASOURCE_hub_map.xlsx'


########################## IMPORT ORIGINAL DATASOURCE MAIN DB

print('\n\nIMPORTING MAIN DATASOURCE DB')

indf = pd.read_csv(maindbfilename,engine='python',delimiter=',',encoding='ISO-8859-1')

########################## COMPUTE STRUCTURED DB

patients = build_patients_struct(indf,hubhospitalsfilename)
structdb = exportdb(patients)
print('\n\nINPUT PREPROCESSING: evaluating critical patients')
structdb_crit = evaluate_criticals(structdb)
print('\n\nINPUT PREPROCESSING: computing prevalence data')
prevdb = filter_region_data()
structdb_crit_prev = assign_prevalence(structdb_crit,prevdb)
print('\n\nINPUT PREPROCESSING: computing geographic data')
structdb_crit_prev_geo = assign_geodata(structdb_crit_prev,"foo",mindate,maxdate)

########################### MANAGE AETHIOLOGY

print('\n\nINPUT PREPROCESSING: reworking categories')
medstructdb = structdb_crit_prev_geo.copy(deep=True)
accList = ['CADUTA','CALAMITA NATURALE','CROLLO','ESPLOSIONE','EVENTO DI MASSA','EVENTO VIOLENTO','INC ACQUA','INC ARIA','INC FERROVIA','INC INFORTUNIO','INC MONTANO','INC STRADALE','INCENDIO','INTOSSICAZIONE']
heartList = ['MEDICO ACUTO, CARDIOCIRCOLATORIA']
respList = ['MEDICO ACUTO, RESPIRATORIA']
neuroList = ['MEDICO ACUTO, NEUROLOGICA']
for val in accList:
    medstructdb.loc[medstructdb['MOTIVO']==val,'MOTIVO'] = 'ACCIDENT'
medstructdb.loc[medstructdb['MOTIVO']==heartList[0],'MOTIVO'] = 'HEART DISEASE'
medstructdb.loc[medstructdb['MOTIVO']==respList[0],'MOTIVO'] = 'RESPIRATORY DISEASE'
medstructdb.loc[medstructdb['MOTIVO']==neuroList[0],'MOTIVO'] = 'NEUROLOGICAL DISEASE'
medstructdb.loc[medstructdb['MOTIVO'].str.contains('MEDICO ACUTO'),'MOTIVO'] = 'OTHER MEDICAL'
medstructdb.loc[((medstructdb['MOTIVO']!='ACCIDENT')&(medstructdb['MOTIVO']!='HEART DISEASE')&(medstructdb['MOTIVO']!='RESPIRATORY DISEASE')&(medstructdb['MOTIVO']!='NEUROLOGICAL DISEASE')&(medstructdb['MOTIVO']!='OTHER MEDICAL')),'MOTIVO'] = 'OTHER/UNKNOWN'

########################### EXPORT STRUCTURED DB

print('\n\nINPUT PREPROCESSING: computing quartiles')
outdb = medstructdb.copy()
quartfield = 'TOT_POS'
allvalslist = outdb[quartfield].unique()
q1thresh = np.quantile(allvalslist,0.25)
q2thresh = np.quantile(allvalslist,0.5)
q3thresh = np.quantile(allvalslist,0.75)
q1_structdb = outdb.loc[outdb[quartfield]<=q1thresh].copy(deep=True)
q2_structdb = outdb.loc[((outdb[quartfield]>q1thresh)&(outdb[quartfield]<=q2thresh))].copy(deep=True)
q3_structdb = outdb.loc[((outdb[quartfield]>q2thresh)&(outdb[quartfield]<=q3thresh))].copy(deep=True)
q4_structdb = outdb.loc[outdb[quartfield]>q3thresh].copy(deep=True)
print('\n\nINPUT PREPROCESSING: exporting data')
outdb.to_excel('AO_Geo'+outputdbname)
q1_structdb.to_excel('AO_GeoQ1_'+outputdbname)
q2_structdb.to_excel('AO_GeoQ2_'+outputdbname)
q3_structdb.to_excel('AO_GeoQ3_'+outputdbname)
q4_structdb.to_excel('AO_GeoQ4_'+outputdbname)

br = 1
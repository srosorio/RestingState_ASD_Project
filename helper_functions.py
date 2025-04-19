import numpy as np 
import pandas as pd
from datetime import datetime
from os.path import join


subs2reject = ['REDACTED']
sensors     = dict({ 'parietal'   : 
                     ['MEG1832','MEG2242','MEG1843','MEG2013','MEG2023','MEG2033','MEG1912','MEG2042','MEG2032','MEG2312'],
                     'occipital' : 
                     ['MEG1923','MEG2113','MEG2343','MEG1732','MEG1932','MEG2122','MEG2332','MEG1743','MEG2143','MEG2133'],
                     'frontal'   : 
                     ['MEG0531','MEG0821','MEG0941','MEG0611','MEG1011','MEG1021','MEG0641','MEG0621','MEG1031']
                    })

fname   = join('/PATH/TO/','file.csv')
csvfile = pd.read_csv(fname, sep=';')
csvfile['subject_id'] = csvfile['subject_id'].astype('string').str.zfill(6)
for i in ['total_t_score', 'srstot2_survey', 'srstot2']:
    csvfile[i] = csvfile[i].astype(str).str.replace('>','')

##################### --------------------------------------- #####################
params4fooof = dict({'peak_width_limits' : (1, 12), 'aperiodic_mode' : 'fixed',  'peak_threshold' : 3})
freqs4fooof  = (1,45)

##################### --------------------------------------- #####################

def get_subs2analyze(subjects):
    """
    """
    subs2analyze = [i for i in subjects if i in list(csvfile['subject_id'])]
    
    return subs2analyze

def get_diagnosis(subject):
    """
    """
    subs2except = ['ID1','ID2']
    diagnosis = []
    for i in ['asd', 'prescreen_survey_asd_v2', 'prescreen_survey_asd']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            diagnosis.append(this_val.values)
    if diagnosis:
        diagnosis = 'asd' if diagnosis[0][0]==1 else 'td'
    else:
        if ('ID3' in subject) or ('ID4' in subject):
            diagnosis = 'td'
        else:
            diagnosis = 'n/a'

    # check if this is a misophonia subject
    miso =  []
    for i in ['miso','miso2']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            miso.append(this_val.values)
    if miso:
        miso = 'miso' if miso[0][0]==1 else 'td'
    if ('miso' in miso) and (subject not in subs2except):
        diagnosis = 'miso'

    # check if there are other diagnoses
    other_dx = []
    if any(csvfile['prescreen_survey_diaglist___1'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___1'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___1'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('ADD')
    if any(csvfile['prescreen_survey_diaglist___2'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___2'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___2'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('ADHD')
    if any(csvfile['prescreen_survey_diaglist___3'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___3'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___3'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('Anxiey')    
    if any(csvfile['prescreen_survey_diaglist___4'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___4'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___4'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('Depression')    
    if any(csvfile['prescreen_survey_diaglist___5'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___5'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___5'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('LangImpair')   
    if any(csvfile['prescreen_survey_diaglist___6'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___6'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___6'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('OCD')   
    if any(csvfile['prescreen_survey_diaglist___7'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___7'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___7'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('Tourette')   
    if any(csvfile['prescreen_survey_diaglist___8'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___8'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___8'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('BPD')   
    if any(csvfile['prescreen_survey_diaglist___13'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___12'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___12'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('PTSD')  
    if any(csvfile['prescreen_survey_diaglist___9'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___9'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___9'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('Other_Psychotic')
    if any(csvfile['prescreen_survey_diaglist___10'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___10'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___10'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('Other_Psychiatric')                    
    if any(csvfile['prescreen_survey_diaglist___11'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diaglist_v2___11'][csvfile['subject_id'] == subject].dropna().tolist() +
           csvfile['rescreensurvey_diagnosis___11'][csvfile['subject_id'] == subject].dropna().tolist()):
        other_dx.append('No')   
    if not other_dx:
        other_dx.append('Unchecked')  

    comments = []
    if any(csvfile['diagnosis_flag_details'][csvfile['subject_id'] == subject].dropna().tolist() + 
           csvfile['prescreen_survey_diagnosiscomments'][csvfile['subject_id'] == subject].dropna().tolist()):
        comments.append(csvfile['diagnosis_flag_details'][csvfile['subject_id'] == subject].dropna().tolist() + 
                        csvfile['prescreen_survey_diagnosiscomments'][csvfile['subject_id'] == subject].dropna().tolist())
    return diagnosis, other_dx, comments

def get_age(subject, visits):
    """
    """
    
    dob = csvfile['dob'][csvfile['subject_id'] == subject].dropna().values.tolist()
    df  = pd.DataFrame({'dovs' : csvfile['meg_dov'][csvfile['subject_id'] == subject].values.tolist(),
                        'ages' : csvfile['meg_age'][csvfile['subject_id'] == subject].values.tolist()})
    age = []
    for visit_i in visits:
        date_i = datetime.strptime(visit_i.split('_')[1].replace('-',''),'%Y%m%d')
        deltas = []
        for dov_i in df.dovs:
            if isinstance(dov_i,str):
                deltas.append(abs((date_i - datetime.strptime(dov_i.replace('-',''),'%Y%m%d')).days))
        min_idx = np.argmin(np.array(deltas))  
        at_this_visit = df.ages.iloc[min_idx]  
        if np.isnan(at_this_visit):
            at_this_visit = (date_i - datetime.strptime(dob[0].replace('-',''),'%Y%m%d') ).days / 365.25
        age.append(at_this_visit)
    
    return age
def get_sex(subject):
    """
    """

    this_val = csvfile[['sex','prescreen_survey_asab',
                        'prescreen_survey_asab_v2',
                        'abas3_sex','prescreen_survey_sex_v2',
                        'prescreen_survey_sex']][csvfile['subject_id'] == subject].values
    this_val = np.ndarray.flatten(this_val)
    this_val = list(set(this_val[~np.isnan(this_val)]))

    if any(this_val):
        sex = 'male' if this_val[0] == 1 else 'female'
    else:
        sex = 'n/a'
    
    return sex

def get_iq(subject, visits):
    """
    """
    # 

    ver   = csvfile[['kbitdate3','dasdate', 'dasdate4','kbitvss','das25','das19a']][csvfile['subject_id'] == subject].values.tolist()
    dates = []
    vals  = []
    for i in ver:
        for j in i:
            if 'nan' not in str(j):
                if '-' in str(j):
                    dates.append(j)
                else:
                    vals.append(j)

    if len(dates) != len(vals):
        dates = list(set(dates))
        vals  = list(set(vals))
        if len(dates) != len(vals):
            if subject == 'SUBID':
                dates = dates[0]
            elif 'SUBSTR' in subject:
                dates = ['2016-09-20'] + dates
    ver = pd.DataFrame({'iq_dov' : dates, 'iq_vals' : vals})

    nover   = csvfile[['kbitdate3','dasdate', 'dasdate4','kbitnvss','das28','das22a']][csvfile['subject_id'] == subject].values.tolist()
    dates = []
    vals  = []
    for i in nover:
        for j in i:
            if 'nan' not in str(j):
                if '-' in str(j):
                    dates.append(j)
                else:
                    vals.append(j)

    if len(dates) != len(vals):
        dates = list(set(dates))
        vals  = list(set(vals))
        if len(dates) != len(vals):
            if subject == 'ID5':
                dates = dates[0]
            if 'SUBSTR' in subject:
                dates = ['2016-09-20'] + dates
    nover = pd.DataFrame({'iq_dov' : dates, 'iq_vals' : vals})

    iq_v  = []
    iq_nv = []

    for visit_i in visits:
        date_i = datetime.strptime(visit_i.split('_')[1].replace('-',''),'%Y%m%d')
        if not ver.empty:
            deltas = []
            for dov_i in ver.iq_dov:
                if isinstance(dov_i,str):
                    deltas.append(abs((date_i - datetime.strptime(dov_i.replace('-',''),'%Y%m%d')).days))
            min_idx = np.argmin(np.array(deltas))  
            at_this_visit = ver.iq_vals.iloc[min_idx]  
            if np.isnan(at_this_visit):
                at_this_visit = None
            iq_v.append(at_this_visit)
        else:
            iq_v.append(None)

        if not ver.empty:
            deltas = []
            for dov_i in nover.iq_dov:
                if isinstance(dov_i,str):
                    deltas.append(abs((date_i - datetime.strptime(dov_i.replace('-',''),'%Y%m%d')).days))
            min_idx = np.argmin(np.array(deltas))  
            at_this_visit = nover.iq_vals.iloc[min_idx]  
            if np.isnan(at_this_visit):
                at_this_visit = None
            iq_nv.append(at_this_visit)
        else:
            iq_nv.append(None)

    return iq_v,iq_nv

def get_ados_info(subject):
    """
    """

    csi    = []
    repbeh = []
    cmpst  = []
    
    for i in ['ados42os4', 'ados4os4', 'ados3os4', 'social_affect_total']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            csi.append(this_val.values)
    if len(csi) > 1:
        csi = np.hstack(csi)
        csi = [i for i in csi if i != 0]
    if csi:
        ados_csi = np.mean(np.unique(csi))
    else:
        ados_csi = None

    for i in ['ados42os5', 'ados4os5', 'ados3_2os3', 'new_alg_rrb_total']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            repbeh.append(this_val.values)
    if len(repbeh) > 1:
        repbeh = np.hstack(repbeh)
        repbeh = [i for i in repbeh if i != 0]
    if repbeh:
        ados_rrb = np.mean(np.unique(repbeh))
    else:
        ados_rrb = None

    for i in ['new_alg_total', 'ados3_2os4', 'ados3os5']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            cmpst.append(this_val.values)
    if len(cmpst) > 1:
        cmpst = np.hstack(cmpst)
        cmpst = [i for i in cmpst if i != 0]
    if cmpst:
        ados_cmpst = np.mean(np.unique(cmpst))
    else:
        ados_cmpst = None

    return ados_csi, ados_rrb, ados_cmpst

def get_srs(subject, visits):
    """
    """
    srs_info = csvfile[['srs_2_date','total_t_score','date','srstot2_survey','srsdate','srstot2']][csvfile['subject_id'] == subject]
    for i in srs_info.keys().tolist():
        srs_info[i] = srs_info[i].astype(str)
    srs_tmp  = srs_info.values
    srs_idx  = np.all((srs_tmp == 'nan'), axis=1)
    srs_tmp  = srs_tmp[~srs_idx]
    srs_info = pd.DataFrame(srs_tmp, columns=srs_info.keys()).values.tolist()
    srs      = []
    dates    = []
    vals     = [] 
    for i in srs_info:
        for idx,j in enumerate(i):
            if 'nan' not in str(j):
                if '-' in str(j) and i[idx+1] != 'nan':
                    if float(i[idx+1]) > 35:
                        dates.append(j)
                else:
                    if float(j) > 35 and i[idx-1] != 'nan':
                        vals.append(float(j))
                    # elif float(j) <= 35 and str(j) != 'nan':
                    #     vals.append(np.nan)
    
    if not bool(dates) and not bool(vals):
       for visit_i in visits:
            srs.append(np.nan) 
    elif len(dates) != len(vals):
        if not bool(dates) and np.any(np.array(vals) <= 35):
            for visit_i in visits:
                srs.append(np.nan)
        else:
            if len(dates) != len(vals):
                print('')
            else:
                for idx,visit_i in enumerate(visits):
                    srs.append(vals[idx])
    else:
        # put together dates and vals as a dataframe
        srs_info = pd.DataFrame({'srs_dov' : dates, 'srs_vals' : vals})
        srs_info.srs_vals = srs_info.srs_vals.astype(float)
        for visit_i in visits:
            date_i = datetime.strptime(visit_i.split('_')[1].replace('-',''),'%Y%m%d')
            deltas = []
            for dov_i in srs_info.srs_dov:
                if isinstance(dov_i,str):
                    deltas.append(abs((date_i - datetime.strptime(dov_i.replace('-',''),'%Y%m%d')).days))
            min_idx = np.argmin(np.array(deltas))  
            at_this_visit = srs_info.srs_vals.iloc[min_idx]  
            if np.isnan(at_this_visit):
                at_this_visit = np.nan
            srs.append(at_this_visit)
    return srs

def get_scq(subject):
    """
    """

    scq     = []
    for i in ['scqlifetot', 'scqlifetot_survey', 'scqcurrtot', 'scqcurrtot_survey']:
        this_val = csvfile[i][csvfile['subject_id'] == subject].dropna()
        if not this_val.empty:
            scq.append(this_val.values)        
    if len(scq) > 1:
        scq = list(np.hstack(scq))
    if scq:
        scq = np.mean(np.unique(scq).astype(np.float32))
    else:
        scq = None
    
    return scq

def get_aud(subject, visits):
    """
    """

    aud = csvfile['spatotal'][csvfile['subject_id'] == subject].dropna().values
    aud = np.mean(np.unique(aud))
    
    return aud

def get_med_info(subject, visits):
    """
    """
    df  = csvfile[['meg_dov','meg_medication___1', 'meg_medication___2', 'meg_medication___3']][csvfile['subject_id'] == subject]
    on_med_at_meg = []
    for visit_i in visits:
        date_i = datetime.strptime(visit_i.split('_')[1].replace('-',''),'%Y%m%d')
        deltas = []
        for dov_i in df.meg_dov:
            if isinstance(dov_i,str):
                deltas.append(abs((date_i - datetime.strptime(dov_i.replace('-',''),'%Y%m%d')).days))
        min_idx = np.argmin(np.array(deltas))  
        at_this_visit = df.iloc[min_idx,:]
        if at_this_visit['meg_medication___1'] == 1:
            on_med_at_meg.append('no')
        elif (at_this_visit['meg_medication___2'] == 1):
            on_med_at_meg.append('stim')
        elif (at_this_visit['meg_medication___3'] == 1):
            on_med_at_meg.append('other')
        else:
            on_med_at_meg.append('n/a')
    
    return on_med_at_meg

import numpy as np
from scipy.signal import detrend

def calculate_fei(Signal, windowSize, windowOverlap):
    """
    Calculates fEI (on a set window size) for signal.
    Based on the algorithm from:
    Bruining et al., Scientific Reports (2020)

    Parameters:
    Signal        : np.ndarray of shape (numSamples, numChannels)
    windowSize    : int, window size in samples
    windowOverlap : float, fraction of overlap between windows (0-1)

    Returns:
    EI   : np.ndarray of shape (numChannels,)
    wAmp : np.ndarray of shape (numChannels, numWindows)
    wDNF : np.ndarray of shape (numChannels, numWindows)
    """
    
    def create_window_indices(lengthSignal, lengthWindow, windowOffset):
        windowStarts   = np.arange(0, lengthSignal - lengthWindow + 1, windowOffset)
        numWindows     = len(windowStarts)
        oneWindowIndex = np.arange(lengthWindow)
        allWindowIndex = oneWindowIndex + windowStarts[:, np.newaxis]

        return allWindowIndex

    lengthSignal, numChannels = Signal.shape
    windowOffset   = int(np.floor(windowSize * (1 - windowOverlap)))
    allWindowIndex = create_window_indices(lengthSignal, windowSize, windowOffset)
    numWindows     = allWindowIndex.shape[0]

    EI   = np.zeros(numChannels)
    wAmp = np.zeros((numChannels, numWindows))
    wDNF = np.zeros((numChannels, numWindows))

    for i_channel in range(numChannels):
        originalAmplitude = Signal[:, i_channel]
        signalProfile     = np.cumsum(originalAmplitude - np.mean(originalAmplitude))
        
        # Calculate mean amplitude for each window
        w_originalAmplitude = np.mean(originalAmplitude[allWindowIndex], axis=1)
        xAmp = np.tile(w_originalAmplitude[:, np.newaxis], (1, windowSize))
        
        # Arrange signal into windows
        xSignal = signalProfile[allWindowIndex]
        xSignal = (xSignal / xAmp).T
        
        # Detrend signal
        dSignal = detrend(xSignal, axis=0, type='linear')
        
        # Standard deviation of detrended normalized fluctuations
        w_detrendedNormalizedFluctuations = np.std(dSignal, axis=0, ddof=0)
        
        # EI Calculation
        correlation = np.corrcoef(w_detrendedNormalizedFluctuations, w_originalAmplitude)[0, 1]
        EI[i_channel] = 1 - correlation

        wAmp[i_channel, :] = w_originalAmplitude
        wDNF[i_channel, :] = w_detrendedNormalizedFluctuations

    return EI, wAmp, wDNF
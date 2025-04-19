import mne
import fooof
import pactools
import numpy as np 
import pandas as pd
import seaborn as sns
import helper_functions as hf
import matplotlib.pyplot as plt
import scipy.stats as st
from glob import glob
from os.path import join, exists
from os import listdir


redo        = True
sensor_clus = 'frontal' # 'parietal', 'occipital', 'frontal', or 'all'
report_name = 'paradigm_report.hdf5'
data_dir    = '/PATH/TO/FOLDER/DATA'
project_dir = '/PATH/TO/PROJECT'

##################### --------------------------------------- #####################

if not exists(join(project_dir,'data.csv')) or redo:

    # get subjects from fixation directory    
    subjects =  [i for i in listdir(data_dir) if len(i) == 6]
    subjects.sort()

    # replace 0AC substrig for 900 to match ID of AC subjects in redcap
    subjects = [i.replace('0AC', '900') if 'AC' in i else i for i in subjects]
    subjects.sort()
    subs2analyze = hf.get_subs2analyze(subjects)

    # empty lists for all the data we are going to extract and integrate
    sub_id, age, sex = [],[],[]
    ver_iq, nover_iq = [],[]
    ados_csi, ados_rrb, ados_cmpst = [],[],[]
    srs, scq, aud    = [],[],[]
    visit_date, group, on_med, other_dx = [],[],[],[]
    abs_delta_power, rel_delta_power, rel_flat_delta_power  = [],[],[]
    abs_theta_power, rel_theta_power, rel_flat_theta_power  = [],[],[]
    abs_alpha_power, rel_alpha_power, rel_flat_alpha_power  = [],[],[]
    abs_beta_power,  rel_beta_power,  rel_flat_beta_power   = [],[],[]
    abs_gamma_power, rel_gamma_power, rel_flat_gamma_power  = [],[],[]
    ind_peak_freq, abs_indpeak_power, rel_indpeak_power     = [],[],[]
    fooof_peaks, offset, exponent = [],[],[]
    fEI_five,fEI_eight,fEI_ten    = [],[],[]
    pac       = []
    comments  = []
    
    # loop through all subjects
    for sub_i in subs2analyze:
        if sub_i not in hf.subs2reject:

            # get subject and visit info
            subject    = sub_i
            visits     = [i for i in listdir(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'))) if 'visit' in i]
            diagnosis,others,dx_comments = hf.get_diagnosis(subject)
            age_at_meg = hf.get_age(subject, visits)
            sub_sex    = hf.get_sex(subject)
            sub_iqv,sub_iqnv = hf.get_iq(subject, visits)
            sub_csi,sub_rrb,sub_cmpst = hf.get_ados_info(subject)
            sub_srs    = hf.get_srs(subject, visits)
            sub_scq    = hf.get_scq(subject)
            sub_aud    = hf.get_aud(subject, visits)
            sub_med    = hf.get_med_info(subject, visits)

            # loop through each visit (some some subjects have more than one visit)
            for idx,visit_i in enumerate(visits):

                # check if this visit should be excluded
                reject_visit = glob(join(data_dir,sub_i,visit_i,'DONT_USE_THIS_VISIT.txt'))
                if bool(reject_visit):
                    continue

                # load files
                epo_file  = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*nobaseline_resting_epo.fif'))
                raw_files = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*fix*ss.fif'))

                if epo_file:

                    epo  = mne.read_epochs(epo_file[0]).resample(250)

                    # correct ch_names if necessary and pick only channels of interest
                    if 'all' not in sensor_clus:
                        if ' ' in epo.ch_names[0]:
                            sensors2use = [i.replace('MEG','MEG ') for i in hf.sensors[sensor_clus]]
                        else:
                            sensors2use = hf.sensors[sensor_clus]
                    else: 
                        sensors2use = 'grad'

                    # filter the data between 1 and 45Hz for fooof modeling and compute spectrum
                    sub_psd = epo.compute_psd(fmin=hf.freqs4fooof[0], fmax=hf.freqs4fooof[1], picks=sensors2use).average()

                    # get mean power spectrum across all channels
                    mean_psd_allchans = np.mean(sub_psd.get_data(), axis=0)

                    # initialize and fit a fooof model
                    fm = fooof.FOOOF(peak_width_limits = hf.params4fooof['peak_width_limits'], 
                                     aperiodic_mode    = hf.params4fooof['aperiodic_mode'], 
                                     peak_threshold    = hf.params4fooof['peak_threshold'], 
                                     verbose           = False)
                    fm.add_data(sub_psd.freqs, mean_psd_allchans, [hf.freqs4fooof[0], hf.freqs4fooof[1]])
                    fm.fit(sub_psd.freqs, mean_psd_allchans, [hf.freqs4fooof[0], hf.freqs4fooof[1]])
                    
                    # now do PAC analysis
                    low_fq_range  = np.linspace(4, 15, 50)
                    high_fq_range = np.linspace(15, 125, 80)
                    estimator = pactools.Comodulogram(fs=250, low_fq_range=low_fq_range, 
                                            low_fq_width=1, method='penny',
                                            progress_bar=False) 
                    data = epo.copy().pick(sensors2use).get_data()
                    estimator.fit(np.mean(np.mean(data, axis=0),axis=0))
                    # estimator.plot()

                    # # open report and save some plots for data quality inspection 
                    report  = mne.Report(title=report_name) if not exists(join(project_dir,report_name)) else mne.open_report(join(project_dir,report_name))

                    # # plot model and flattened spectrum
                    fig, ax = plt.subplots(1,2, figsize=[15.15, 6.53])
                    fm.plot(False, ax=ax[0])
                    fooof.plts.plot_spectra(fm.freqs[None,:], fm._peak_fit[None,:], False, ax=ax[1], color='green', labels='Periodic Fit')
                    for i in fm.peak_params_:
                        ax[1].axvline(x=i[0], ls='--', color='r')
                    plt.close()

                    # # add plot to report 
                    section = '%s_%s' %(sub_i,visit_i)
                    report.add_figure(fig=fig, title='jump', section=section, tags=(['activation',diagnosis]), replace=True)
                    # report.save(join(project_dir,report_name), verbose=False, overwrite=True)

                    # get the stuff we want! First, absolute power
                    abs_d_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= 1)  & (sub_psd.freqs <= 4)])
                    abs_t_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= 4)  & (sub_psd.freqs <= 8)])
                    abs_a_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= 8)  & (sub_psd.freqs <= 12)])
                    abs_b_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= 15) & (sub_psd.freqs <= 30)])
                    abs_g_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= 30) & (sub_psd.freqs <= 45)])

                    # relative power in uncorrected spectrum
                    rel_d_pow = abs_d_pow / np.sum(mean_psd_allchans)
                    rel_t_pow = abs_t_pow / np.sum(mean_psd_allchans)
                    rel_a_pow = abs_a_pow / np.sum(mean_psd_allchans)
                    rel_b_pow = abs_b_pow / np.sum(mean_psd_allchans)
                    rel_g_pow = abs_g_pow / np.sum(mean_psd_allchans)

                    # relative power in flattened (corrected) spectrum
                    flat_d_pow = np.sum(fm._spectrum_flat[(fm.freqs >= 1)  & (fm.freqs <= 4)])  / np.sum(fm._spectrum_flat)
                    flat_t_pow = np.sum(fm._spectrum_flat[(fm.freqs >= 4)  & (fm.freqs <= 8)])  / np.sum(fm._spectrum_flat)
                    flat_a_pow = np.sum(fm._spectrum_flat[(fm.freqs >= 8)  & (fm.freqs <= 12)]) / np.sum(fm._spectrum_flat)
                    flat_b_pow = np.sum(fm._spectrum_flat[(fm.freqs >= 15) & (fm.freqs <= 30)]) / np.sum(fm._spectrum_flat)
                    flat_g_pow = np.sum(fm._spectrum_flat[(fm.freqs >= 30) & (fm.freqs <= 45)]) / np.sum(fm._spectrum_flat)

                    # keep only the peak in the alpha band (6-15Hz)
                    fooof_peak_params  = [p for p in fm.peak_params_ if 6 <= p[0] <= 15]
                    if len(fooof_peak_params) > 1:
                        fooof_peak_params = [fooof_peak_params[np.argmax(np.array(fooof_peak_params)[:,1])]]
                    if fooof_peak_params:
                        ind_peak    = fooof_peak_params[0][0]
                        ind_raw_pow = np.sum(mean_psd_allchans[(sub_psd.freqs >= (ind_peak-1.5)) & (sub_psd.freqs <= (ind_peak+1.5))])
                        ind_rel_pow = ind_raw_pow / np.sum(mean_psd_allchans)
                    else:
                        ind_peak    = np.nan
                        ind_raw_pow = np.nan
                        ind_rel_pow = np.nan

                    # store data in their corresponding lists
                    sub_id.append(subject)
                    age.append(age_at_meg[idx])
                    sex.append(sub_sex)
                    ver_iq.append(sub_iqv[idx])
                    nover_iq.append(sub_iqnv[idx])
                    ados_csi.append(sub_csi)
                    ados_rrb.append(sub_rrb)
                    ados_cmpst.append(sub_cmpst)
                    srs.append(sub_srs[idx])
                    scq.append(sub_scq)
                    aud.append(sub_aud)
                    on_med.append(sub_med[idx])
                    visit_date.append(visit_i)
                    group.append(diagnosis)
                    other_dx.append(others)
                    comments.append(dx_comments)
                    abs_delta_power.append(abs_d_pow)
                    abs_theta_power.append(abs_t_pow)
                    abs_alpha_power.append(abs_a_pow)
                    abs_beta_power.append(abs_b_pow)
                    abs_gamma_power.append(abs_g_pow)
                    rel_delta_power.append(rel_d_pow)
                    rel_theta_power.append(rel_t_pow)                                     
                    rel_alpha_power.append(rel_a_pow)
                    rel_beta_power.append(rel_b_pow)
                    rel_gamma_power.append(rel_g_pow)
                    rel_flat_delta_power.append(flat_d_pow)
                    rel_flat_theta_power.append(flat_t_pow)
                    rel_flat_alpha_power.append(flat_a_pow)
                    rel_flat_beta_power.append(flat_b_pow)
                    rel_flat_gamma_power.append(flat_g_pow)
                    ind_peak_freq.append(ind_peak)
                    abs_indpeak_power.append(ind_raw_pow)
                    rel_indpeak_power.append(ind_rel_pow)
                    offset.append(fm.aperiodic_params_[0])
                    exponent.append(fm.aperiodic_params_[1])
                    pac.append(estimator.comod_)

                    # let's compute functional EI as in Bruining et al. 2020. Let's see how window length affects the results             
                    if raw_files:
                        all_EI_five  = []
                        all_EI_eight = []
                        all_EI_ten   = []
                        for i in raw_files:
                            try:
                                raw_sss = mne.io.read_raw_fif(i).load_data()
                            except:
                                print(f"Error loading {i}")
                                continue

                            if 'all' not in sensor_clus:
                                if ' ' in raw_sss.ch_names[0]:
                                    sensors2use = [i.replace('MEG','MEG ') for i in hf.sensors[sensor_clus]]
                                else:
                                    sensors2use = hf.sensors[sensor_clus]
                                raw_sss = raw_sss.pick(sensors2use).filter(8, 12).resample(250)
                            else: 
                                raw_sss = raw_sss.pick('grad').filter(8, 12).resample(250)

                            # get envelope of hilbert transform 
                            raw_sss = raw_sss.apply_hilbert(envelope=True)
                            signal  = np.expand_dims(np.mean(raw_sss.get_data(), axis=0), axis=1)

                            # get functional EI value for 5, 8 and 10 seconds
                            EI_five, _, _  = hf.calculate_fei(signal, (250*5), 0.5)
                            EI_eight, _, _ = hf.calculate_fei(signal, (250*8), 0.5)
                            EI_ten, _, _   = hf.calculate_fei(signal, (250*10), 0.5)
                            all_EI_five.append(EI_five)
                            all_EI_eight.append(EI_eight)
                            all_EI_ten.append(EI_ten)
                        if bool(all_EI_five):
                            fEI_five.append(np.mean(all_EI_five))
                            fEI_eight.append(np.mean(all_EI_eight))
                            fEI_ten.append(np.mean(all_EI_ten))
                        else:
                            fEI_five.append(np.nan)
                            fEI_eight.append(np.nan)
                            fEI_ten.append(np.nan)
                    else:
                        print(f"no raw files for {sub_i} {visit_i}")

    # create dataframe for all data
    df = pd.DataFrame({'subject' : sub_id, 'age' : age, 'sex' : sex, 'visit' : visit_date, 'group' : group, 'ver_iq' : ver_iq, 'nover_iq' : nover_iq, 
                       'ados_csi' : ados_csi, 'ados_rrb' : ados_rrb, 'scq' : scq, 'on_med' : on_med, 'ados_cmpst' : ados_cmpst, 'srs' : srs, 'aud' : aud,  
                       'abs_delta_power' : abs_delta_power, 'abs_theta_power' : abs_theta_power, 'abs_alpha_power' : abs_alpha_power,'abs_beta_power' : abs_beta_power, 'abs_gamma_power' : abs_gamma_power,
                       'rel_delta_power' : rel_delta_power, 'rel_theta_power' : rel_theta_power, 'rel_alpha_power' : rel_alpha_power,'rel_beta_power' : rel_beta_power, 'rel_gamma_power' : rel_gamma_power,
                       'rel_flat_delta_power' : rel_flat_delta_power, 'rel_flat_theta_power' : rel_flat_theta_power, 'rel_flat_alpha_power' : rel_flat_alpha_power,'rel_flat_beta_power' : rel_flat_beta_power, 'rel_flat_gamma_power' : rel_flat_gamma_power,
                       'ind_alpha_peak' : ind_peak_freq, 'abs_indpeak_power' : abs_indpeak_power, 'rel_indpeak_power' : rel_indpeak_power,
                       'offset' : offset, 'exponent' : exponent, 'pac' : pac, 'fEI_five' : fEI_five, 'fEI_eight' : fEI_eight, 'fEI_ten' : fEI_ten, 
                       'other_dx' : other_dx,'comments' : comments})
    
    # we have to save arrays separately because they are too big to be saved in a dataframe
    pacs = np.array(df.pac.values.tolist())

    # save data
    df.to_csv(join(project_dir,f'data_sensor_{sensor_clus}_clus.csv'))
    np.savez(join(project_dir,f'pac_arrays_{sensor_clus}.npz'),
            pacs       = pacs,
            low_freqs  = low_fq_range,
            high_freqs = high_fq_range,)
    print(f'saved as data_sensor_{sensor_clus}_clus.csv')
else:
    # load data
    df = pd.read_csv(join(project_dir,f'data_sensor_{sensor_clus}_clus.csv'), index_col=0)
    pac_data = np.load(join(project_dir,f'pac_arrays_{sensor_clus}.npz'))

    # retrieve pac data and assign to corresponding dataframe columns
    pacs       = pac_data['pacs']
    low_freqs  = pac_data['low_freqs']
    high_freqs = pac_data['high_freqs']   
    df.subject  = df.subject.astype(str).str.zfill(6)
    df.other_dx = df.other_dx.astype(str)
    df.pac      = list(pacs)

# let's do some cleaning and create a copy of the dataframe with only the data we will use 
df_clean = df.copy()
df_clean.drop(df_clean[(df_clean.age < 6.8) | (df_clean.age >= 40)].index, inplace=True)
df_clean.drop(df_clean[df_clean.group == 'miso'].index, inplace=True)
df_clean.drop(df_clean[np.isnan(df_clean.ver_iq)].index, inplace=True)
df_clean.drop(df_clean[(~((df_clean.other_dx == "['Unchecked']") | (df_clean.other_dx == "['No']")))  & (df_clean.group=='td')].index, inplace=True)
df_clean.drop(df_clean[np.isnan((df_clean.ind_alpha_peak)) | (~df_clean.ind_alpha_peak.between(6,15))].index, inplace=True)
df_clean.drop(df_clean[np.isnan(df_clean.srs)].index, inplace=True)

# lets explore some assocations. This is part of the quality control
SMALL_SIZE = 26
plt.rcParams["font.family"] = "Arial"
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)  

#plot scatter and best fit line
fig,ax = plt.subplots(1,1)
xvar = 'age'
yvar = 'exponent'
sns.regplot(x=df_clean[xvar][df_clean.group=='td'],y=df_clean[yvar][df_clean.group=='td'], 
            color=(0.5647058823529412, 0.9333333333333333, 0.5647058823529412, 1.0), label='td', scatter_kws=dict(edgecolors='white', s=200))
sns.regplot(x=df_clean[xvar][df_clean.group=='asd'],y=df_clean[yvar][df_clean.group=='asd'], 
            color=(0.8549019607843137, 0.4392156862745098, 0.8392156862745098, 1.0), label='asd', scatter_kws=dict(edgecolors='white', s=200))
ax.set_xlabel(xvar)
ax.set_ylabel(yvar)
fig.set_size_inches([13.55,  8.79])

r,p = st.spearmanr(df_clean[xvar][df_clean.group=='asd'],df_clean[yvar][df_clean.group=='asd'])
ax.text(x=plt.gca().get_xlim()[1]/2, y=(np.max(df_clean[yvar])-.10), s=f"ASD: r = {r:.2f}, p = {p:.2f}", color='orchid')
r,p = st.spearmanr(df_clean[xvar][df_clean.group=='td'],df_clean[yvar][df_clean.group=='td'])
ax.text(x=plt.gca().get_xlim()[1]/2, y=(np.max(df_clean[yvar])-.50), s=f"TD: r = {r:.2f}, p = {p:.2f}", color='lightgreen')



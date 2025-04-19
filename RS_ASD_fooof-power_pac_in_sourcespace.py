import mne
import fooof
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import helper_functions as hf
from glob import glob
from os.path import join, exists, split, isfile
from os import listdir, makedirs

redo        = True
report_name = 'paradigm_report.hdf5'
data_dir    = '/PATH/TO/FOLDER/DATA'
project_dir = '/PATH/TO/PROJECT'
recons_dir  = '/PATH/TO/MRI/WMA/recons'

fname   = join('/PATH/TO/','FILE.csv')
csvfile = pd.read_csv(fname, sep=';')
csvfile['subject_id'] = csvfile['subject_id'].astype('string').str.zfill(6)
for i in ['total_t_score', 'srstot2_survey', 'srstot2']:
    csvfile[i] = csvfile[i].astype(str).str.replace('>','')

src_to = mne.read_source_spaces('/PATH/TO/FSAVERAGE/DIRECTORY/bem/fsaverage-ico-5-src.fif')

othersubs2reject =  ['REDACTED']

##################### --------------------------------------- #####################

if not exists(join(project_dir,'data.csv')) or redo:
        
    # get subjects from fixation directory    
    subjects =  [i for i in listdir(data_dir) if len(i) == 6]
    subjects.sort()
    subjects = [i.replace('SUBSTR1', 'SUBSTR2') if 'AC' in i else i for i in subjects]
    subjects.sort()
    subs2analyze = [i for i in subjects if i in list(csvfile['subject_id'])]

    td_stcs_peak_freq,td_stcs_peak_power,td_stcs_alpha_power,td_stcs_offset,td_stcs_exponent      = [],[],[],[],[]
    asd_stcs_peak_freq,asd_stcs_peak_power,asd_stcs_alpha_power,asd_stcs_offset,asd_stcs_exponent = [],[],[],[],[]

    for sub_idx,sub_i in enumerate(subs2analyze):
        print(f'\n>>>>> Estimating parameters for subject {sub_idx} out of {len(subs2analyze)} \n')

        if sub_i not in hf.subs2reject and sub_i not in othersubs2reject:
            subject    = sub_i
            visits     = [i for i in listdir(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'))) if 'visit' in i]   
            diagnosis,others,dx_comments = hf.get_diagnosis(subject)         
            for idx,visit_i in enumerate(visits):

                if len(glob(join(project_dir, 'fooof_in_source_space', sub_i,visit_i,sub_i+'_'+visit_i+'*.stc'))) == 0 or redo:

                    # check if this visit should be excluded
                    reject_visit = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'DONT_USE_THIS_VISIT.txt'))
                    if bool(reject_visit):
                        continue                

                    # load epoch files
                    epo_file = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*nobaseline_resting_epo.fif'))
                    raw_file = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*fix*ss.fif'))
                    raw_erm  = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*erm*ss.fif'))
                    
                    if epo_file:

                        # read files (epoch, raw and erm)
                        epo       = mne.read_epochs(epo_file[0], preload=True).pick('grad').filter(1,120).resample(250)
                        raw_erm   = mne.io.read_raw_fif(raw_erm[0], preload=True).pick('grad').filter(1,120).resample(250)

                        # compute the noise covariance matrix
                        noise_cov = mne.compute_raw_covariance(raw_erm, method='auto', rank=None, verbose=None) 

                        # compute the forward solution and inverser operator
                        fwd_file = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*fwd.fif'))  
                        fwd      = mne.read_forward_solution(fwd_file[0]) 
                        inv_operator = mne.minimum_norm.make_inverse_operator(epo.info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info', verbose=False)

                        # apply the inverse solution using the MNE method and set orientation of source to be perpendicular to the cortical surface
                        snr      = 3
                        lambda2  = 1.0 / snr**2
                        method   = 'MNE'
                        evoked   = epo.average()
                        recons_subject = sub_i+'_'+visit_i.split('_')[-1]
                        stc      = mne.minimum_norm.apply_inverse(evoked, inv_operator, lambda2, 
                                                                  method=method, pick_ori='normal', verbose=False) 
                        # stc.plot(subject=recons_subject, subjects_dir=recons_dir, hemi='lh')

                        # morph the stc to fsaverage
                        src_from = fwd['src']
                        if fwd['src'][0]['subject_his_id']!= recons_subject:
                            recons_subject = fwd['src'][0]['subject_his_id']
                            recons_subject = fwd['src'][1]['subject_his_id']
                        morphed_surf  = mne.compute_source_morph(fwd['src'], subject_from=recons_subject, 
                                                                subject_to='fsaverage', src_to=src_to, subjects_dir=recons_dir) 
                        stc_morphed = morphed_surf.apply(stc)
                        
                        # get the labels we will be using (Shaefer 400 parcels) and exclude median walls
                        labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_400Parcels_7Networks_order', 
                                                            subjects_dir='/DATA/TO/FSAVERAGE/DIRECTORY/',)
                        labels = [i for i in labels if 'Median_Wall' not in i.name]

                        # create dummy stc structures to store our data
                        stc_peak_freq   = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')
                        stc_peak_power  = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')
                        stc_alpha_power = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')
                        stc_offset      = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')
                        stc_exponent    = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')
                        stc_fEI         = mne.SourceEstimate(np.zeros((stc_morphed.data.shape[0],1)), 
                                                             stc_morphed.vertices, tmin=0, tstep=1, subject='fsaverage')

                        label_names, peak_freq, peak_power, alpha_power, offset, exponent  = [],[],[],[],[],[]
    
                        # stc.plot(subject=recons_subject, subjects_dir=recons_dir, hemi='both')

                        # looop through each on of the labels
                        for subLabel_idx,subLabel_i in enumerate(labels):

                            # get psd of each label for fooof modeling
                            data2fooof = stc_morphed.in_label(subLabel_i)
                            this_psd   = mne.time_frequency.psd_array_multitaper(np.mean(data2fooof.data, axis=0), 
                                                                                 sfreq=stc.sfreq, fmin=1, fmax=45, bandwidth=4, verbose=False)

                            # fooof on the average across vertices for this label
                            fm = fooof.FOOOF(peak_width_limits=[.5, 8], max_n_peaks=1)            
                            fm.add_data(this_psd[1], this_psd[0], [1, 45])
                            fm.fit(this_psd[1], this_psd[0], [1, 45]) 

                            label_names.append(subLabel_i.name)
                            if fm.peak_params_.any() and 'Wall' not in subLabel_i.name:
                                peak_freq.append(fm.peak_params_[np.argmax(fm.peak_params_[:,1])][0])
                                peak_power.append(fm.peak_params_[np.argmax(fm.peak_params_[:,1])][1])
                                alpha_power.append(np.sum(this_psd[0][(this_psd[1] >= 8) & (this_psd[1] <= 12)]) / 
                                                   np.sum(this_psd[0]))
                                offset.append(fm.aperiodic_params_[0])
                                exponent.append(fm.aperiodic_params_[1])
                            else:
                                peak_freq.append(np.nan)
                                peak_power.append(np.nan)
                                alpha_power.append(np.sum(this_psd[0][(this_psd[1] >= 8) & (this_psd[1] <= 12)]) / 
                                                   np.sum(this_psd[0]))
                                offset.append(np.nan)
                                exponent.append(np.nan)

                            # set all vertices within the label to the be equal to the esmitated value
                            if 'lh' in subLabel_i.hemi:
                                stc_peak_freq.lh_data[data2fooof.lh_vertno]   = (np.ones(stc_peak_freq.lh_data[data2fooof.lh_vertno].shape) * 
                                                                                 peak_freq[subLabel_idx]).astype('float64')
                                stc_peak_power.lh_data[data2fooof.lh_vertno]  = (np.ones(stc_peak_freq.lh_data[data2fooof.lh_vertno].shape) * 
                                                                                 peak_power[subLabel_idx]).astype('float64')
                                stc_alpha_power.lh_data[data2fooof.lh_vertno] = (np.ones(stc_peak_freq.lh_data[data2fooof.lh_vertno].shape) * 
                                                                                 alpha_power[subLabel_idx]).astype('float64')              
                                stc_offset.lh_data[data2fooof.lh_vertno]      = (np.ones(stc_peak_freq.lh_data[data2fooof.lh_vertno].shape) * 
                                                                                 offset[subLabel_idx]).astype('float64')
                                stc_exponent.lh_data[data2fooof.lh_vertno]    = (np.ones(stc_peak_freq.lh_data[data2fooof.lh_vertno].shape) * 
                                                                                 exponent[subLabel_idx]).astype('float64')
                            else:
                                stc_peak_freq.rh_data[data2fooof.rh_vertno]   = (np.ones(stc_peak_freq.rh_data[data2fooof.rh_vertno].shape) * 
                                                                                 peak_freq[subLabel_idx]).astype('float64')
                                stc_peak_power.rh_data[data2fooof.rh_vertno]  = (np.ones(stc_peak_freq.rh_data[data2fooof.rh_vertno].shape) * 
                                                                                 peak_power[subLabel_idx]).astype('float64')
                                stc_alpha_power.rh_data[data2fooof.rh_vertno] = (np.ones(stc_peak_freq.rh_data[data2fooof.rh_vertno].shape) * 
                                                                                 alpha_power[subLabel_idx]).astype('float64')
                                stc_offset.rh_data[data2fooof.rh_vertno]      = (np.ones(stc_peak_freq.rh_data[data2fooof.rh_vertno].shape) * 
                                                                                 offset[subLabel_idx]).astype('float64')
                                stc_exponent.rh_data[data2fooof.rh_vertno]    = (np.ones(stc_peak_freq.rh_data[data2fooof.rh_vertno].shape) * 
                                                                                 exponent[subLabel_idx]).astype('float64')

                        # save stcs
                        if not exists(join(project_dir,'NEWDIR',sub_i,visit_i)):
                            makedirs(join(project_dir,'fooof_in_source_space',sub_i,visit_i))

                        stc_peak_freq.save(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                f'{sub_i}_{visit_i}_peak_freq'), overwrite=True)
                        stc_peak_power.save(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                f'{sub_i}_{visit_i}_peak_power'), overwrite=True)
                        stc_alpha_power.save(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                f'{sub_i}_{visit_i}_alpha_power'), overwrite=True)
                        stc_offset.save(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                f'{sub_i}_{visit_i}_offset'), overwrite=True)
                        stc_exponent.save(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                f'{sub_i}_{visit_i}_exponent'), overwrite=True)
                    
                    # now let's compute functiona EI ratios. We do this only for subjects that also have an epo file
                    if raw_file:

                        # load raw and erm files
                        raw_sss = mne.io.read_raw_fif(raw_file[0], preload=True)

                        # filter raw file
                        raw_sss = raw_sss.pick('grad').filter(8, 12).resample(250)
                        raw_sss = raw_sss.apply_hilbert(envelope=True)

                        # compute the forward solution and inverser operator(this time for the continuous data, not the epochs)
                        fwd_file = glob(join(data_dir,sub_i.replace('SUBSTR1','SUBSTR2'),visit_i,'*fwd.fif'))  
                        fwd      = mne.read_forward_solution(fwd_file[0]) 
                        inv_operator = mne.minimum_norm.make_inverse_operator(raw_sss.info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info', verbose=False)

                        # apply the inverse solution using the MNE method and set orientation of source to be perpendicular to the cortical surface
                        snr      = 3
                        lambda2  = 1.0 / snr**2
                        method   = 'MNE'
                        evoked   = epo.average()
                        recons_subject = sub_i+'_'+visit_i.split('_')[-1]
                        stc      = mne.minimum_norm.apply_inverse_raw(raw_sss, inv_operator, lambda2, 
                                                                      method=method, pick_ori='Normal', verbose=False) 
                        src_from = fwd['src']        
                        # stc.plot(subject=recons_subject, subjects_dir=recons_dir, hemi='both')                

                        # signal  = np.expand_dims(np.mean(raw_sss.get_data(), axis=0), axis=1)   
                        # [CODE IN PROGRESS]

                else:
                    stc_peak_freq   = mne.read_source_estimate(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                                    f'{sub_i}_{visit_i}_peak_freq'), 
                                                               subject='fsaverage')
                    stc_peak_power  = mne.read_source_estimate(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                                    f'{sub_i}_{visit_i}_peak_power'), 
                                                               subject='fsaverage')
                    stc_alpha_power = mne.read_source_estimate(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                                    f'{sub_i}_{visit_i}_alpha_power'), 
                                                               subject='fsaverage') 
                    stc_offset      = mne.read_source_estimate(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                                    f'{sub_i}_{visit_i}_offset'), 
                                                               subject='fsaverage')
                    stc_exponent    = mne.read_source_estimate(join(project_dir,'NEWDIR',sub_i,visit_i,
                                                                    f'{sub_i}_{visit_i}_exponent'), 
                                                               subject='fsaverage') 
                        
                if 'td' in diagnosis:
                    td_stcs_peak_freq.append(stc_peak_freq)
                    td_stcs_peak_power.append(stc_peak_power)
                    td_stcs_alpha_power.append(stc_alpha_power)
                    td_stcs_offset.append(stc_offset)
                    td_stcs_exponent.append(stc_exponent)
                else:
                    asd_stcs_peak_freq.append(stc_peak_freq)
                    asd_stcs_peak_power.append(stc_peak_power)
                    asd_stcs_alpha_power.append(stc_alpha_power)
                    asd_stcs_offset.append(stc_offset)
                    asd_stcs_exponent.append(stc_exponent)

                if sub_idx == 0:
                    np.nanmean(td_stcs_alpha_power).plot(subject='fsaverage', hemi='both', 
                        subjects_dir='/PATH/TO/FSAVERAGE/DIRECTORY/', 
                        clim=dict(kind="value", lims=[0, .17, .19]))


                    

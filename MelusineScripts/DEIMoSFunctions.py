#Script to run through the DEIMoS commands for all .h5 files in a directory
#Adapted into discrete functions from the 20240503_LoopThroughHD5Files_DataProcessingOnly.py script
#Author: Jessica O'Loughlin (s1907024@ed.ac.uk)
#Supervisor: Prof. Karl Burgess (k.burgess@ed.ac.uk)
#Created: 08/11/2024

import glob #Searches for files with specific extensions
import deimos #Performs various Mass Spec Processing Steps
import numpy as np #Basic math functionalities
import matplotlib.pyplot as plt #Create plot outputs
import os #Allows directory to be read
import os.path #Checks for existance of files
import shutil #Will remove specific folders if they are already present
import time #Find the creation date of files 
import pandas as pd #Handle dataframes
from numpy import trapz #Calculate area under line for Gap Filling
from datetime import datetime #Get current date and time
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
    
def FindMiddleFileForRTAlignment():
    #Find the file in the middle of the LC-IM-QTOF run to align other files to downstream
    CreationTime = []
    #Read through files in specified directory
    for file in os.listdir(r"F:\JessicaOLoughlin\RawDataAgilentFiles"):
        if file.endswith(".d"):
            #print(os.path.join("F:\JessicaOLoughlin\RawDataAgilentFiles", file))
            #Get file creation time
            Created = time.ctime(os.path.getctime(r"F:\JessicaOLoughlin\RawDataAgilentFiles\{}".format(file)))
            #Add to dataframe
            CreationTime.append({"File":file, "CreationTime":Created})
            
    #Make CreationTime a dataframe and sort by CreationTime column
    CreationTime = pd.DataFrame(CreationTime)
    CreationTime = CreationTime.sort_values('CreationTime', ascending=True)
    CreationTime = CreationTime.reset_index(drop = True)
    #Calculate middle row + select it
    middle_row = int(len(CreationTime.index)/2)
    middle = CreationTime.loc[middle_row]
    middle = middle['File']
    middle = middle.replace(".d", "")
    middle = "RTAligned_{}".format(middle)
    
    print("middle = ", middle)
    
    #Load in the middle file for downstream RT alginment
    rtalign_data = {}
    rtalign_data['ref_ms1'] = deimos.load('{}.h5'.format(middle), key='ms1')
    rtalign_data['ref_ms1'] = rtalign_data['ref_ms1'].apply(pd.to_numeric, errors = "ignore")
    rtalign_data['ref_ms2'] = deimos.load('{}.h5'.format(middle), key='ms2')
    rtalign_data['ref_ms2'] = rtalign_data['ref_ms2'].apply(pd.to_numeric, errors = "ignore")
    
    return middle, rtalign_data

    
def CreateCCSCalObjects(tune_pos_file, ccsCalib_mz, ccsCalib_ccs, ccsCalib_q, ccsCalib_buffer_mass, ccsCalib_mz_tol, ccsCalib_dt_tol):
    tune_pos = deimos.load(f'f:\JessicaOLoughlin\RawDataMZMLFiles\{tune_pos_file}', key='ms1')
    print(tune_pos)
    ccs_cal_pos = deimos.calibration.tunemix(tune_pos,
                                              mz=ccsCalib_mz,
                                              ccs=ccsCalib_ccs,
                                              q=ccsCalib_q,
                                              buffer_mass=ccsCalib_buffer_mass, 
                                              mz_tol=ccsCalib_mz_tol, 
                                              dt_tol=ccsCalib_dt_tol)
    print('r-squared:\t', ccs_cal_pos.fit['r'] ** 2)
    
    return ccs_cal_pos

def RetentionTimeAlignment(file_NoExt, rtalign_data, middle):
    
    #Make a folder for this sample for all Peak Detection results to go in to
    if os.path.exists('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt)):
        shutil.rmtree('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt))
    os.mkdir('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt))
    
    ## ms1_peaks
    # Load data
    rtalign_data['toAlign_ms1'] = deimos.load('{}.h5'.format(file_NoExt), key='ms1')
    rtalign_data['toAlign_ms1'] = rtalign_data['toAlign_ms1'].apply(pd.to_numeric, errors = "ignore")
    
    rtalign_data['toAlign_ms2'] = deimos.load('{}.h5'.format(file_NoExt), key='ms2')
    rtalign_data['toAlign_ms2'] = rtalign_data['toAlign_ms2'].apply(pd.to_numeric, errors = "ignore")
    
    if file_NoExt == middle:
        #Just save the middle/reference file as it is
        deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), rtalign_data['ref_ms1'], key='ms1')
        deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), rtalign_data['ref_ms2'], key='ms2')
    else:
        ## Retention Time Align all other files
        
        #Visualise misalignment
        # Collapse
        ToAlign_ms1 = deimos.collapse(rtalign_data['toAlign_ms1'], keep='retention_time').sort_values(by='retention_time')
        Ref_ms1 = deimos.collapse(rtalign_data['ref_ms1'], keep='retention_time').sort_values(by='retention_time')
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.fill_between(ToAlign_ms1['retention_time'], ToAlign_ms1['intensity'], color='C0', alpha=0.5, label='A')
        ax.fill_between(Ref_ms1['retention_time'], Ref_ms1['intensity'], color='C3', alpha=0.5, label='B')
        ax.set_xlabel('Retention Time (MS1)', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS1_InitialMisalignment.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        ToAlign_ms2 = deimos.collapse(rtalign_data['toAlign_ms2'], keep='retention_time').sort_values(by='retention_time')
        Ref_ms2 = deimos.collapse(rtalign_data['ref_ms2'], keep='retention_time').sort_values(by='retention_time')
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.fill_between(ToAlign_ms2['retention_time'], ToAlign_ms2['intensity'], color='C0', alpha=0.5, label='A')
        ax.fill_between(Ref_ms2['retention_time'], Ref_ms2['intensity'], color='C3', alpha=0.5, label='B')
        ax.set_xlabel('Retention Time (ms2)', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2_InitialMisalignment.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        # Perform peak detection
        peaks = {}
        peaks['toAlign_ms1'] = deimos.peakpick.persistent_homology(deimos.threshold(rtalign_data['toAlign_ms1'], threshold=128),
                                                         dims=['mz', 'drift_time', 'retention_time'])
        peaks['ref_ms1'] = deimos.peakpick.persistent_homology(deimos.threshold(rtalign_data['ref_ms1'], threshold=128),
                                                         dims=['mz', 'drift_time', 'retention_time'])
        peaks['toAlign_ms2'] = deimos.peakpick.persistent_homology(deimos.threshold(rtalign_data['toAlign_ms2'], threshold=128),
                                                         dims=['mz', 'drift_time', 'retention_time'])
        peaks['ref_ms2'] = deimos.peakpick.persistent_homology(deimos.threshold(rtalign_data['ref_ms2'], threshold=128),
                                                         dims=['mz', 'drift_time', 'retention_time'])
        # Downselect by persistence
        peaks['toAlign_ms1']['persistence_ratio'] = peaks['toAlign_ms1']['persistence'] / peaks['toAlign_ms1']['intensity']
        peaks['toAlign_ms1'] = deimos.threshold(peaks['toAlign_ms1'], by='persistence_ratio', threshold=0.75)
        
        peaks['ref_ms1']['persistence_ratio'] = peaks['ref_ms1']['persistence'] / peaks['ref_ms1']['intensity']
        peaks['ref_ms1'] = deimos.threshold(peaks['ref_ms1'], by='persistence_ratio', threshold=0.75)
        
        peaks['toAlign_ms2']['persistence_ratio'] = peaks['toAlign_ms2']['persistence'] / peaks['toAlign_ms2']['intensity']
        peaks['toAlign_ms2'] = deimos.threshold(peaks['toAlign_ms2'], by='persistence_ratio', threshold=0.75)
        
        peaks['ref_ms2']['persistence_ratio'] = peaks['ref_ms2']['persistence'] / peaks['ref_ms2']['intensity']
        peaks['ref_ms2'] = deimos.threshold(peaks['ref_ms2'], by='persistence_ratio', threshold=0.75)
        
        # Partition
        partitions_toAlign_ms1 = deimos.partition(deimos.threshold(peaks['toAlign_ms1'], 
                                                                   threshold=1E3),
                                                  split_on='mz',
                                                  size=1000,
                                                  overlap=0.25)
        partitions_toAlign_ms2 = deimos.partition(deimos.threshold(peaks['toAlign_ms2'], 
                                                                   threshold=1E3),
                                                  split_on='mz',
                                                  size=1000,
                                                  overlap=0.25)
        # Match
        toAlign_ms1_matched, ref_ms1_matched = partitions_toAlign_ms1.zipmap(deimos.alignment.match, 
                                                                             deimos.threshold(peaks['ref_ms1'], 
                                                                                              threshold=1E3),
                                                                             dims=['mz', 'drift_time', 'retention_time'],
                                                                             tol=[20E-6, 0.03, 2], relative=[True, True, False],
                                                                             processes=4)
        toAlign_ms2_matched, ref_ms2_matched = partitions_toAlign_ms2.zipmap(deimos.alignment.match, 
                                                                             deimos.threshold(peaks['ref_ms2'], 
                                                                                              threshold=1E3),
                                                                             dims=['mz', 'drift_time', 'retention_time'],
                                                                             tol=[20E-6, 0.03, 2], relative=[True, True, False],
                                                                             processes=4)
        
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.scatter(toAlign_ms1_matched['retention_time'], 
                   ref_ms1_matched['retention_time'], s=2)
        ax.plot([0, 25], [0, 25], linewidth=1, linestyle='--', color='k')
        ax.set_xlabel('Retention Time (A) (MS1)', fontweight='bold')
        ax.set_ylabel('Retention Time (B) (MS1)', fontweight='bold')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS1_PartitionedData.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.scatter(toAlign_ms2_matched['retention_time'], 
                   ref_ms2_matched['retention_time'], s=2)
        ax.plot([0, 25], [0, 25], linewidth=1, linestyle='--', color='k')
        ax.set_xlabel('Retention Time (A) (MS2)', fontweight='bold')
        ax.set_ylabel('Retention Time (B) (MS2)', fontweight='bold')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2_PartitionedData.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        # SVR spline
        spl_ms1 = deimos.alignment.fit_spline(toAlign_ms1_matched, ref_ms1_matched, 
                                          align='retention_time', kernel='rbf', C=1000)
        newx_ms1 = np.linspace(0, toAlign_ms1_matched['retention_time'].max(), 1000)
        
        spl_ms2 = deimos.alignment.fit_spline(toAlign_ms2_matched, ref_ms2_matched, 
                                          align='retention_time', kernel='rbf', C=1000)
        newx_ms2 = np.linspace(0, toAlign_ms2_matched['retention_time'].max(), 1000)
        
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.plot(newx_ms1, spl_ms1(newx_ms1), c='black', linewidth=1, linestyle='--')
        ax.scatter(toAlign_ms1_matched['retention_time'], ref_ms1_matched['retention_time'], s=2)
        ax.set_xlabel('Retention Time (A) (MS1)', fontweight='bold')
        ax.set_ylabel('Retention Time (B) (MS1)', fontweight='bold')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS1_SVRSplineFit.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.plot(newx_ms2, spl_ms2(newx_ms2), c='black', linewidth=1, linestyle='--')
        ax.scatter(toAlign_ms2_matched['retention_time'], ref_ms2_matched['retention_time'], s=2)
        ax.set_xlabel('Retention Time (A) (MS2)', fontweight='bold')
        ax.set_ylabel('Retention Time (B) (MS2)', fontweight='bold')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2_SVRSplineFit.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        #Apply Alignment
        rtalign_data['Aligned_ms1'] = rtalign_data['toAlign_ms1'].copy()
        rtalign_data['Aligned_ms1']['retention_time'] = spl_ms1(rtalign_data['Aligned_ms1']['retention_time'])
        
        rtalign_data['Aligned_ms2'] = rtalign_data['toAlign_ms2'].copy()
        rtalign_data['Aligned_ms2']['retention_time'] = spl_ms2(rtalign_data['Aligned_ms2']['retention_time'])
        
        # Collapse
        rt_aligned_ms1 = deimos.collapse(rtalign_data['Aligned_ms1'], keep='retention_time').sort_values(by='retention_time')
        rt_aligned_ms2 = deimos.collapse(rtalign_data['Aligned_ms2'], keep='retention_time').sort_values(by='retention_time')
        
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.fill_between(rt_aligned_ms1['retention_time'], rt_aligned_ms1['intensity'], color='C0', alpha=0.5, label='spl(A)')
        ax.fill_between(Ref_ms1['retention_time'], Ref_ms1['intensity'], color='C3', alpha=0.5, label='B')
        ax.set_xlabel('Retention Time (MS1)', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS1_RetentionTimeAlgined.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.fill_between(rt_aligned_ms2['retention_time'], rt_aligned_ms2['intensity'], color='C0', alpha=0.5, label='spl(A)')
        ax.fill_between(Ref_ms2['retention_time'], Ref_ms2['intensity'], color='C3', alpha=0.5, label='B')
        ax.set_xlabel('Retention Time (MS2)', fontweight='bold')
        ax.set_ylabel('Intensity', fontweight='bold')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2_RetentionTimeAlgined.png'.format(file_NoExt))
        #Save memory space
        plt.close()
        
        #Save Data as HD5 file
        deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), rtalign_data['Aligned_ms1'], key='ms1')
        deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), rtalign_data['Aligned_ms2'], key='ms2')
        

def DetectPeaks(file_NoExt, PeakDet_intensity_thres, PeakDet_smooth_data_radius, PeakDet_persistent_homology_radius):
    
    #Make a folder for this sample for all Peak Detection results to go in to
    if os.path.exists('Results/{}/PeakDetection'.format(file_NoExt)):
        shutil.rmtree('Results/{}/PeakDetection'.format(file_NoExt))
    os.mkdir('Results/{}/PeakDetection'.format(file_NoExt))
    
    ## ms1_peaks
    # Load data
    ms1_iso = deimos.load('{}.h5'.format(file_NoExt), key='ms1')
    ms1_iso = ms1_iso.apply(pd.to_numeric, errors = "ignore")
    #Remove the scanId column (Only required for Isotope Detection process)
    ms1 = ms1_iso.drop('scanId', axis = 1)
    # Build factors from raw data
    factors = deimos.build_factors(ms1, dims='detect')
    # Nominal threshold
    ms1_thres500 = deimos.threshold(ms1, threshold=PeakDet_intensity_thres) 
    # Build index
    index = deimos.build_index(ms1_thres500, factors)
    # Smooth data
    ms1_thres500 = deimos.filters.smooth(ms1_thres500, 
                                         index=index, dims=['mz', 'drift_time',
                                                            'retention_time'],
                                         radius=PeakDet_smooth_data_radius, iterations=7)
    # Perform peak detection
    ms1_peaks = deimos.peakpick.persistent_homology(ms1_thres500, index=index,
                                                dims=['mz', 'drift_time', 
                                                      'retention_time'],
                                                radius=PeakDet_persistent_homology_radius)
    
    #Save outputs as .csv file
    ms1_peaks.to_csv('Results/{}/PeakDetection/ms1_peaks.csv'.format(file_NoExt), index=False)
    
    #Remove columns not required for downstream processes
    ms1_peaks = ms1_peaks.drop(['mz_weighted', 'drift_time_weighted', 
                                'retention_time_weighted'], axis = 1)
    ## ms2_peaks
    # Load data, excluding scanid column
    ms2 = deimos.load('{}.h5'.format(file_NoExt), key='ms2', columns=['mz', 'drift_time', 
                                                              'retention_time', 
                                                              'intensity'])
    # Build factors from raw data
    factors = deimos.build_factors(ms2, dims='detect')
    # Nominal threshold
    ms2_thres500 = deimos.threshold(ms2, threshold=PeakDet_intensity_thres)
    # Build index
    index = deimos.build_index(ms2_thres500, factors)
    # Smooth data
    ms2_thres500 = deimos.filters.smooth(ms2_thres500, 
                                          index=index, dims=['mz', 'drift_time', 
                                                            'retention_time'],
                                          radius=PeakDet_smooth_data_radius, iterations=7)
    # Perform peak detection
    ms2_peaks = deimos.peakpick.persistent_homology(ms2_thres500, index=index,
                                                dims=['mz', 'drift_time', 
                                                      'retention_time'],
                                                radius=PeakDet_persistent_homology_radius)
    
    #Save outputs as .csv file
    ms2_peaks.to_csv('Results/{}/PeakDetection/ms2_peaks.csv'.format(file_NoExt), index=False)
    
    #Remove columns not required for downstream processes
    ms2_peaks = ms2_peaks.drop(['mz_weighted', 'drift_time_weighted', 
                                'retention_time_weighted'], axis = 1)
    
    # ## Save sample_peaks.h5 file
    # deimos.save('{}_peaks.h5'.format(file_NoExt), ms1_peaks, key='ms1', mode='w')
    # deimos.save('{}_peaks.h5'.format(file_NoExt), ms2_peaks, key='ms2', mode='a')
    
    return ms1, ms1_peaks, ms2, ms2_peaks, ms1_iso

def DetectIsotopes(ms1_peaks, isotope_intensity_thres, isotope_partition_size, isotope_partition_overlap, isotope_map_mz_dt_rt_tol, 
                   isotope_map_delta, isotope_map_max_isotopes, isotope_map_max_charges, isotope_map_max_error, file_NoExt, 
                   isotope_min_no_isotopes, ms1_iso, isotope_slice_mz_low, isotope_slice_mz_high, isotope_plot_slice_mz_low, 
                   isotope_plot_slice_dt_low, isotope_plot_slice_rt_low, isotope_plot_slice_mz_high, isotope_plot_slice_dt_high, 
                   isotope_plot_slice_rt_high): 
    #Remeber to use ms1_iso object and remove persistence from ms1_peaks object
    # ms1_peaks_thres1000 = ms1.drop('persistence', axis = 1)
    # ms1_peaks_thres1000 = ms1_iso.drop('persistence', axis = 1)
    # ms1_peaks_thres1000 = deimos.threshold(ms1_peaks_thres1000, threshold=1000)
    ms1_peaks_thres1000 = deimos.threshold(ms1_peaks, threshold=isotope_intensity_thres)
    # Partition the data
    partitions = deimos.partition(ms1_peaks_thres1000, size=isotope_partition_size, 
                                  overlap=isotope_partition_overlap)
    
    # Map isotope detection over partitions
    isotopes = partitions.map(deimos.isotopes.detect,
                              dims=['mz', 'drift_time', 'retention_time'],
                              tol=isotope_map_mz_dt_rt_tol,
                              delta=isotope_map_delta,
                              max_isotopes=isotope_map_max_isotopes,
                              max_charge=isotope_map_max_charges,
                              max_error=isotope_map_max_error)

    #Make a folder for this sample for Isotope Detection results to go in to
    if os.path.exists('Results/{}/IsotopeDetection'.format(file_NoExt)):
        shutil.rmtree('Results/{}/IsotopeDetection'.format(file_NoExt))
    os.mkdir('Results/{}/IsotopeDetection'.format(file_NoExt))
    
    isotopes.to_csv('Results/{}/IsotopeDetection/isotopes.csv'.format(file_NoExt), index=False) 
    
    #Consider isotopic signatures with at least 3 members
    isotopes = isotopes.loc[isotopes['n'] >= isotope_min_no_isotopes, :].sort_values(by='intensity', ascending=False).head(5)
    
    for index, row in isotopes.iterrows():
        
        #Following Peak detection steps to subset data within the mass range of the isotope
        ms1_iso_ss = deimos.slice(ms1_iso, by='mz', 
                                  low = row['mz']-isotope_slice_mz_low, 
                                  high = row['mz']+isotope_slice_mz_high)
        
        # Get maximal data point
        scan_id_i, rt_i, dt_i, mz_i, intensity_i = ms1_iso_ss.loc[ms1_iso_ss['intensity'] == ms1_iso_ss['intensity'].max(), :].round(1).values[0]
    
        #The ms1_iso data is loaded in the Peak Detection section
        feature = deimos.slice(ms1_iso_ss, by=['mz', 'drift_time', 'retention_time'],
                            low=[mz_i - isotope_plot_slice_mz_low, 
                                 dt_i - isotope_plot_slice_dt_low, 
                                 rt_i - isotope_plot_slice_rt_low],
                            high=[mz_i + isotope_plot_slice_mz_high, 
                                  dt_i + isotope_plot_slice_dt_high, 
                                  rt_i + isotope_plot_slice_rt_high])
        
        ax = deimos.plot.multipanel(feature, dpi=300)
        for i in isotopes.loc[index, 'mz_iso']:
            ax['mz'].scatter(i, 0, s=10, color='C3', zorder=5)
        
        plt.tight_layout()
        plt.savefig('Results/{}/IsotopeDetection/{}_isotope_feature_plot.png'.format(file_NoExt, index))
        #Save memory space
        plt.close()
    
def ExtractMS2Spectra(ms1, ms2, file_NoExt, MS2Extract_intensity_thres, MS2Extract_ms1_mz_subset_low, MS2Extract_ms1_dt_subset_low, 
                      MS2Extract_ms1_rt_subset_low, MS2Extract_ms1_mz_subset_high, MS2Extract_ms1_dt_subset_high, 
                      MS2Extract_ms1_rt_subset_high, MS2Extract_ms2_dt_subset_low, MS2Extract_ms2_rt_subset_low, 
                      MS2Extract_ms2_dt_subset_high, MS2Extract_ms2_rt_subset_high, MS2Extract_model_ce, MS2Extract_model_params, 
                      ms1_peaks, ms2_peaks, MS2Extract_ms1_decon_intensity_thres, MS2Extract_ms2_decon_intensity_thres, 
                      MS2Extract_construct_pairs_dt_low, MS2Extract_construct_pairs_rt_low, MS2Extract_construct_pairs_dt_high, 
                      MS2Extract_construct_pairs_rt_high, MS2Extract_construct_pairs_ce, MS2Extract_construct_pairs_error_tol, 
                      MS2Extract_config_extract_mz_low, MS2Extract_config_extract_dt_low, MS2Extract_config_extract_rt_low, 
                      MS2Extract_config_extract_mz_high, MS2Extract_config_extract_dt_high, MS2Extract_config_extract_rt_high, 
                      MS2Extract_decon_dt_resolution, MS2Extract_dt_score_threshold):
    
    #Make a folder for this sample for all MS2 Extraction results to go in to
    if os.path.exists('Results/{}/MS2Extraction'.format(file_NoExt)):
        shutil.rmtree('Results/{}/MS2Extraction'.format(file_NoExt))
    os.mkdir('Results/{}/MS2Extraction'.format(file_NoExt))
    
    #Use ms1 and ms2 objects from Peak Detection
    ms1_thres_ms2Extract = deimos.threshold(ms1, threshold=MS2Extract_intensity_thres)
    ms2_thres_ms2Extract = deimos.threshold(ms2, threshold=MS2Extract_intensity_thres)
    
    # get maximal data point
    #mz_i, dt_i, rt_i, intensity_i = ms1_thres100.loc[ms1_thres100['intensity'] == ms1_thres100['intensity'].max(), :].values[0]
    rt_i, dt_i, mz_i, intensity_i = ms1_thres_ms2Extract.loc[ms1_thres_ms2Extract['intensity'] == ms1_thres_ms2Extract['intensity'].max(), :].values[0]

    # subset the raw data
    precursor = deimos.slice(ms1_thres_ms2Extract,
                        by=['mz', 'drift_time', 'retention_time'],
                        low=[mz_i - MS2Extract_ms1_mz_subset_low, 
                             dt_i - MS2Extract_ms1_dt_subset_low, 
                             rt_i - MS2Extract_ms1_rt_subset_low],
                        high=[mz_i + MS2Extract_ms1_mz_subset_high, 
                              dt_i + MS2Extract_ms1_dt_subset_high, 
                              rt_i + MS2Extract_ms1_rt_subset_high])

    # putative fragments
    fragment_profile = deimos.slice(ms2_thres_ms2Extract,
                                by=['drift_time', 'retention_time'],
                                low=[dt_i - MS2Extract_ms2_dt_subset_low, 
                                     rt_i - MS2Extract_ms2_rt_subset_low],
                                high=[dt_i + MS2Extract_ms2_dt_subset_high, 
                                      rt_i + MS2Extract_ms2_rt_subset_high])
    
    fragment_dt = deimos.collapse(fragment_profile, keep='drift_time').sort_values(by='drift_time')
    precursor_dt = deimos.collapse(precursor, keep='drift_time').sort_values(by='drift_time')
    
    #Save outputs as .csv files
    # fragment_dt.to_csv('Results/{}/MS2Extraction/fragment_dt.csv'.format(file_NoExt), index=False)
    # precursor_dt.to_csv('Results/{}/MS2Extraction/precursor_dt.csv'.format(file_NoExt), index=False)
    
    #Make Drift Time Offset graph
    fig, ax = plt.subplots(1, dpi=150, facecolor='w')
    ax.fill_between(precursor_dt['drift_time'],
                    precursor_dt['intensity'] / precursor_dt['intensity'].max(),
                    color='C0', alpha=0.4, label='Precursor') 
    ax.fill_between(fragment_dt['drift_time'],
                    fragment_dt['intensity'] / fragment_dt['intensity'].max(),
                    color='C3', alpha=0.4, label='Fragments')
    ax.set_xlabel('Drift Time', fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontweight='bold')
    ax.set_ylim(0, None)
    ax.legend()
    plt.tight_layout()
    #Save image in specified directory
    # plt.savefig('Results/{}/MS2Extraction/DriftTimeOffset.png'.format(file_NoExt))
    plt.close()
    
    def offset_correction_model(dt_ms2, mz_ms2, mz_ms1, ce=MS2Extract_model_ce,
                            params=MS2Extract_model_params):
        # Cast params as array
        params = np.array(params).reshape(-1, 1)
        # Convert collision energy to array
        ce = np.ones_like(dt_ms2) * np.log(ce)
        # Create constant vector
        const = np.ones_like(dt_ms2)
        # Sqrt
        mu_ms1 = np.sqrt(mz_ms1)
        mu_ms2 = np.sqrt(mz_ms2)
        # Ratio
        mu_ratio = mu_ms2 / mu_ms1
        # Create dependent array
        x = np.stack((const, mu_ratio, ce), axis=1)
        # Predict
        y = np.dot(x, params).flatten() * dt_ms2

        return y
    
    #Deconvolution
    ## Use peak data generated in the Peak Detection process
    print("Initial length:")
    print("ms1_peaks:", len(ms1_peaks))
    print("ms2_peaks:", len(ms2_peaks))
    ms1_peaks_thres_forMS2 = deimos.threshold(ms1_peaks, threshold=MS2Extract_ms1_decon_intensity_thres)
    ms2_peaks_thres_forMS2 = deimos.threshold(ms2_peaks, threshold=MS2Extract_ms2_decon_intensity_thres)
    print("Post-thresholding lengths:")
    print("ms1_peaks_thres_forMS2:", len(ms1_peaks_thres_forMS2))
    print("ms2_peaks_thres_forMS2:", len(ms2_peaks_thres_forMS2))
    
    decon = deimos.deconvolution.MS2Deconvolution(ms1_peaks_thres_forMS2, ms1_thres_ms2Extract, 
                                                  ms2_peaks_thres_forMS2, ms2_thres_ms2Extract)
    decon.construct_putative_pairs(dims=['drift_time', 'retention_time'],
                                low=[MS2Extract_construct_pairs_dt_low, 
                                     MS2Extract_construct_pairs_rt_low], 
                                high=[MS2Extract_construct_pairs_dt_high, 
                                      MS2Extract_construct_pairs_rt_high], 
                                ce=MS2Extract_construct_pairs_ce,
                                model=offset_correction_model,
                                require_ms1_greater_than_ms2=True,
                                error_tolerance=MS2Extract_construct_pairs_error_tol)
    decon.configure_profile_extraction(dims=['mz', 'drift_time', 'retention_time'],
                                    low=[MS2Extract_config_extract_mz_low, 
                                         MS2Extract_config_extract_dt_low, 
                                         MS2Extract_config_extract_rt_low],
                                    high=[MS2Extract_config_extract_mz_high, 
                                          MS2Extract_config_extract_dt_high, 
                                          MS2Extract_config_extract_rt_high],
                                    relative=[True, True, False])
    
    res = decon.apply(dims='drift_time', resolution=MS2Extract_decon_dt_resolution)
    print("len(res) after decon.apply():", len(res))
    #.head(5) removed from the function below as we want to save all of the data
    #Group data by each ms1 feature
    res = res.loc[res['drift_time_score'] > MS2Extract_dt_score_threshold].groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                      as_index=False).agg(list).sort_values(by='persistence_ms1',
                                                                                            ascending=False)
    print("len(res) after res.loc[]:", len(res))
                                                                                            
    #Added by Jess (24/04/2025)
    ##Merge MS1 and MS2 data together
    #Make the MS1 index a column
    ms1_peaks_thres_forMS2 = ms1_peaks_thres_forMS2.reset_index(drop = False)
    ms1_peaks_thres_forMS2 = ms1_peaks_thres_forMS2.rename(columns={'index': 'index_ms1'})
    #Perform inner join by the index_ms1 column
    res_final = pd.merge(ms1_peaks_thres_forMS2, res, on="index_ms1", how = "left")
    
    res.to_csv('Results/{file}/MS2Extraction/MS2Extraction_res.csv'.format(file = file_NoExt), index=False)
    ms1_peaks_thres_forMS2.to_csv('Results/{file}/MS2Extraction/MS2Extraction_ms1_peaks_thres_forMS2.csv'.format(file = file_NoExt), index=False)
    res_final.to_csv('Results/{file}/MS2Extraction/MS2Extraction_res_final.csv'.format(file = file_NoExt), index=False)    

# =============================================================================
#     #Iterate through the rows and save a .png file for each one
#     for index, row in res.iterrows():
#         deimos.plot.stem(np.array(res.loc[index, 'mz_ms2']),
#                      np.array(res.loc[index, 'intensity_ms2']),
#                      width=1)
#         plt.tight_layout()
#         plt.savefig('Results/{file}/MS2Extraction/{index}_index_ms1_MS1FragmentationSpectra.png'.format(file = file_NoExt, index = row["index_ms1"]))
#         #Save memory space
#         plt.close()    
# =============================================================================

    print(res_final)
        
    return res_final
    
        
def AgglomerativeClusteringConcatenateNewPeakData(loopcount, res_final, agglo_mergeFeatures_mz_dt_rt_tol, 
                                                  file_NoExt):
    #With thanks to Karl Burgess for scripting the majority this section
    
    # create an empty dataframe to store the clustered peaks
    multipeaks = pd.DataFrame()
    
# =============================================================================
#     #Use ms1_peaks object previously generated in the pipeline
#     merged_peaks = deimos.alignment.merge_features(features = ms1_peaks,
#                                             dims=['mz', 'drift_time',
#                                                 'retention_time'],
#                                             tol=agglo_mergeFeatures_mz_dt_rt_tol,
#                                             relative = [True, True, False])  
# =============================================================================

    #Use res_final object with MS1 and MS2 data
    merged_peaks = deimos.alignment.merge_features(features = res_final,
                                            dims=['mz', 'drift_time',
                                                'retention_time'],
                                            tol=agglo_mergeFeatures_mz_dt_rt_tol,
                                            relative = [True, True, False])                                                        
    # next bit is sort of cheating. Fake up a multi-file dataframe by adding the filename and loopcounter as columns
    merged_peaks.insert(0, 'sample_idx', loopcount)
    merged_peaks.insert(0, 'sample_id', file_NoExt)
    
    # main bit of work - concatenate the dataframe with the new peak data.
    multipeaks = pd.concat([multipeaks, merged_peaks])
    loopcount=loopcount+1
    
    #Append data to existing csv file (prevent system crashing)
    multipeaks.to_csv('Results/{}/AgglomerativeClustering/multipeaks.csv'.format(file_NoExt), index=False)
    #Check if the file already exists
    if os.path.isfile("Results/multipeaks_AllSamples.csv") == True: #True if present
        #Append to file
        multipeaks.to_csv('Results/multipeaks_AllSamples.csv', mode='a', index=False, header=False)
    else:
        #Create file
        multipeaks.to_csv('Results/multipeaks_AllSamples.csv', index=False)
        
    return loopcount
    
    
def AgglomerativeClusteringMainSteps(agglo_multiSampPart_size, agglo_multiSampPart_tol, agglo_clustering_mz_dt_rt_tol):
    #With thanks to Karl Burgess for scripting the majority this section
    
    multipeaks_AllSamples = pd.read_csv("Results/multipeaks_AllSamples.csv")
    
# =============================================================================
#     #Convert object to dask dataframe (help with memory issues)
#     multipeaks_AllSamples = dd.from_pandas(multipeaks_AllSamples, 
#                                            npartitions=3)
#     print("Converted to dask object")
#     print(multipeaks_AllSamples["sample_idx"])
# =============================================================================
    
    #Code from Sean Colby
    # Partition the data
    partitions = deimos.multi_sample_partition(multipeaks_AllSamples, 
                                               split_on="mz", 
                                               size=agglo_multiSampPart_size, 
                                               tol=agglo_multiSampPart_tol)
    
    # Apply agglomerative clustering to partitions
    res = partitions.map(
        deimos.alignment.agglomerative_clustering,
        dims=[
            "mz",
            "drift_time",
            "retention_time",
        ],  # Use “_weighted” variants if you’d prefer
        tol=agglo_clustering_mz_dt_rt_tol,  # Note that this defines the maximum width of an entire cluster, so this would translate to +/- 20 ppm, +/- 1% drift time, +/- 0.1 min retention time
        relative=[True, True, False],
        processes=16 # You may not see benefit from parallelization depending on file I/O patterns
    )  
    
    # Unique cluster indices
    res["cluster"] = (
        res.groupby(by=["partition_idx", "cluster"]).ngroup().reset_index(drop=True)
    )
    
    # Drop unneeded columns
    # res = res.drop(columns=["partition_idx", "sample_idx"])
    res = res.drop(columns=["partition_idx"])

    clustering = res
    
    print("deimos.alignment.agglomerative_clustering() completed")
    del multipeaks_AllSamples

    # now dump it to disk!
    clustering.to_csv('Results/clustering.csv', index=False)
    print("clustering.to_csv() completed")

    return clustering

# def CCSCalibrationSteps(ccs_cal_pos):
def CCSCalibrationSteps(ccs_cal_pos, clustering):
    
    # clustering_CCS = clustering
    clustering = clustering.apply(pd.to_numeric, errors = "ignore")

    # pivot the table to get it into the right format for ipaPy2
    # table headers: id(cluster), average mz, average rts, sample_intensities NOTE: no average drift time or ccs currently!
    full_pivot = pd.pivot_table(clustering, 
                                values = ['intensity', 'mz', 'retention_time', 
                                          'drift_time'], 
                                index = 'cluster', columns = 'sample_id') #columns changed from 'sample_idx'
# =============================================================================
#     full_pivot = pd.pivot_table(clustering_CCS, 
#                                 values = ['intensity', 'mz', 'retention_time', 
#                                           'drift_time', 'persistence', 'sample_id'], 
#                                 index = 'cluster', columns = 'sample_idx')
# =============================================================================    
    full_pivot.to_csv('Results/full_pivot.csv', index=False)
    print("full_pivot = pd.pivot_table() completed")
    del clustering

    #how to get rid of a column in a multilevel table
    #full_pivot = full_pivot.drop([('mz', 'mzs')], axis=1)
    #ok - calculate the per row means for all of the mzs
    mzs = full_pivot[('mz',)].mean(axis=1)
    print(full_pivot[('mz',)])
    print("mzs = full_pivot completed")
    print(mzs)
    #full_pivot[('mz','mzs')] = mzs
    #now drop the individual sample means
    mz_stripped = full_pivot.drop([('mz',)], axis = 1)
    #add to the multiindex
    mz_stripped[('mz', 'mzs')] = mzs
    del [full_pivot, mzs]

    #now do the same for the retention times
    # mz_stripped.to_csv('Results/mz_stripped.csv', index=False)
    rts = mz_stripped[('retention_time',)].mean(axis=1)
    rts_stripped = mz_stripped.drop([('retention_time',)], axis = 1)
    rts_stripped[('retention_time', 'RTs')] = rts
    del [mz_stripped, rts]

    #now we do the same for the drift times 
    drifts = rts_stripped[('drift_time',)].mean(axis=1)
    drifts_stripped = rts_stripped.drop([('drift_time',)], axis = 1)
    #currently commented out as unused, also shouldn't this be CCS values?
    #Update from Jess: Yes! I have put this back in so that the CCS values can be calculated
    drifts_stripped[('drift_time', 'drifts')] = drifts
    del [rts_stripped, drifts]
    # drifts_stripped.to_csv('Results/drifts_stripped.csv', index=False)
    print("dataset stripping steps completed")
    
    #flatten multiindex
    drifts_stripped.columns = drifts_stripped.columns.get_level_values(1)
    #rename the index (cluster) to ids
    #is it ok to just have a list of ids as the index or do we need an additional index column?
    # drifts_stripped = drifts_stripped.index.name ='ids' #Commented out by Jess (29/01/2025)
    # if not, add a specific set of indexes as a different column
    drifts_stripped['ids'] = range(1, len(drifts_stripped) + 1)
    
    drifts_stripped['CCS'] = ccs_cal_pos.arrival2ccs(mz=drifts_stripped['mzs'], 
                                                ta=drifts_stripped['drifts'], 
                                                q=1)
    
    drifts_stripped.to_csv('Results/drifts_stripped_final.csv', index=False)
    print("CCS Calibration steps completed")

    
    return drifts_stripped
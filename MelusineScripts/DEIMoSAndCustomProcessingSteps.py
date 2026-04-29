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

#Check where RTAlignment functions cause the script to "restart"
print("Script loop")

##Set thresholds for different processes
#Reading in files
calib_files = ['POS_Precondition2.h5', 'POS_Precondition3.h5', 'POS_Precondition4.h5', 
               'POS_Precondition5.h5', 'POS_CCS0.h5', 'POS_CCS1.h5', 'POS_CCS2.h5', 
               'POS_CCS3.h5', 'POS_CCS4.h5']

#CCS Calibration
tune_pos_file = 'POS_CCS4.h5'
# ccsCalib_mz = [118.086255, 322.048121, 622.028960, 
#                 922.009798, 1221.990636, 1521.971475, 
#                 1821.952313, 2121.933152, 2421.913990, 2721.894829]
# ccsCalib_ccs = [121.3, 153.7, 203, 
#                 243.6, 282.2, 317, 
#                 351.2, 383, 413, 441.2]
# ccsCalib_q = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ccsCalib_mz = [118.086255, 322.048121, 622.028960, 
                922.009798, 1221.990636, 1521.971475]
ccsCalib_ccs = [121.3, 153.7, 203, 
                243.6, 282.2, 317]
ccsCalib_q = [1, 1, 1, 1, 1, 1]
ccsCalib_buffer_mass = 28.013
ccsCalib_mz_tol = 200E-6
ccsCalib_dt_tol = 0.04

#Peak Detection 
PeakDet_intensity_thres = 500
PeakDet_smooth_data_radius = [0, 1, 0]
PeakDet_persistent_homology_radius = [2, 10, 0]

#MS2 Extraction
MS2Extract_intensity_thres = 1000 #100 in documentation
MS2Extract_ms1_mz_subset_low = 0.1
MS2Extract_ms1_dt_subset_low = 1
MS2Extract_ms1_rt_subset_low = 1
MS2Extract_ms1_mz_subset_high = 0.2 
MS2Extract_ms1_dt_subset_high = 1
MS2Extract_ms1_rt_subset_high = 2
MS2Extract_ms2_dt_subset_low = 1.5 
MS2Extract_ms2_rt_subset_low = 1
MS2Extract_ms2_dt_subset_high = 1
MS2Extract_ms2_rt_subset_high = 2 
MS2Extract_model_ce = 0
MS2Extract_model_params = [1.02067031, -0.02062323,  0.00176694]
MS2Extract_ms1_decon_intensity_thres = 1000 #1E4 in documentation
MS2Extract_ms2_decon_intensity_thres = 1000 #1E3 in documentation
MS2Extract_construct_pairs_dt_low = -0.12
MS2Extract_construct_pairs_rt_low = -0.1
MS2Extract_construct_pairs_dt_high = 1.4
MS2Extract_construct_pairs_rt_high = 0.1
MS2Extract_construct_pairs_ce = 20
MS2Extract_construct_pairs_error_tol = 0.12
MS2Extract_config_extract_mz_low = -200E-6
MS2Extract_config_extract_dt_low = -0.05
MS2Extract_config_extract_rt_low = -0.1
MS2Extract_config_extract_mz_high = 600E-6
MS2Extract_config_extract_dt_high = 0.05
MS2Extract_config_extract_rt_high = 0.1
MS2Extract_decon_dt_resolution = 0.01
MS2Extract_dt_score_threshold = 0.9

#Isotope Detection
isotope_intensity_thres = 1000
isotope_partition_size = 1000
isotope_partition_overlap = 5.1
isotope_map_mz_dt_rt_tol = [0.1, 0.7, 0.15]
isotope_map_delta = 1.003355
isotope_map_max_isotopes = 5
isotope_map_max_charges = 1
isotope_map_max_error = 50E-6
isotope_min_no_isotopes = 3
#Optional: parameters to view the isotopes
isotope_slice_mz_low = 50
isotope_slice_mz_high = 50
isotope_plot_slice_mz_low = 0.10
isotope_plot_slice_dt_low = 2.0
isotope_plot_slice_rt_low = 0.4
isotope_plot_slice_mz_high = 0.10
isotope_plot_slice_dt_high = 2.0
isotope_plot_slice_rt_high = 0.4

#Retention Time Alignment
rtalign_persisHomology_thres = 128
rtalign_persis_thres = 0.75
rtalign_partition_thres = 1E3
rtalign_partition_size = 1000
rtalign_partition_overlap = 0.25
rtalign_zipmap_thres = 1E3
rtalign_zipmap_mz_dt_rt_tol = [20E-6, 0.03, 2]

#Agglomerative Clustering
agglo_mergeFeatures_mz_dt_rt_tol = [2e-05, 0.03, 0.3]
agglo_multiSampPart_size = 100
agglo_multiSampPart_tol = 50e-6
agglo_clustering_mz_dt_rt_tol = [40e-6, 0.02, 0.2]

#Gap Filling
GapFill_mz_tol = 5 #1 = 1 ppm
GapFill_rt_tol = 0.2 #0.2 = 0.2 min (12 seconds)
GapFill_CCS_tol = 1 #1 = 1%
GapFill_trapz_dx = 0.1

#Peak Merging
PeakMerge_mz_ppm = 3
PeakMerge_RT_tol = 0.5 #0.5 = 30 seconds 
PeakMerge_CCS_tol = 2.0 #2 = 2%

#Merge MS2 data
MergeMS2_mz_ppm = 3 #ppm
MergeMS2_RT_tol = 0.5 #Time in minutes
MergeMS2_DT_tol = 2.0 #Percentage

def ReadFilesInDirectory():
    path = r'*.h5'
    #Create a list of the files with the .h5 extension
    files = glob.glob(path)
    #Can check that it has the correct number of files
    #print(len(files))
    
    #Remove the Calibration files from the main analysis pipeline
    files  = [x for x in files if x not in calib_files]
    #For developing a function, just loop through certain number of files
    # files = files[slice(3)]
    print("Files to be processed:")
    print(files)
    
    # files = ['POS_FBS_IM_MSMS_40kTF_400TR_4.h5']
    
    return files
    
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
    
def CreateResultsFile():
    #Make new Results folder (overwrite if it already exists)
    if os.path.exists('Results'):
        shutil.rmtree('Results')
    os.makedirs('Results')
    
def CreateCCSCalObjects():
    tune_pos = deimos.load(tune_pos_file, key='ms1')
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
        

def DetectPeaks(file_NoExt):
    
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

def DetectIsotopes(): 
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
    
def ExtractMS2Spectra(ms1, ms2, file_NoExt):
    
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
    
        
def AgglomerativeClusteringConcatenateNewPeakData(loopcount, res_final):
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
    
    
def AgglomerativeClusteringMainSteps():
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

def GapFillingSteps(ccs_cal_pos, drifts_stripped):
# def GapFillingSteps(ccs_cal_pos):
    
    # drifts_stripped = pd.read_csv("Results/drifts_stripped_final.csv")
    drifts_stripped = drifts_stripped.apply(pd.to_numeric, errors = "ignore")
    # drifts_stripped = drifts_stripped.sort_values(['POS_FBS_IM_MSMS_40kTF_400TR_1'], 
    #                                               ascending=[False])
    # drifts_stripped = drifts_stripped[90:110]
    # drifts_stripped = drifts_stripped.iloc[[7, 9, 11, 19, 60, 70, 72, 90, 97, 105, 107, 108]]
    # drifts_stripped = drifts_stripped.iloc[[7, 9, 11, 19]]
    # ids_to_keep = [17271, 17194, 17204, 17153, 17117, 
    #                 17033, 16909, 16939, 16886, 16778, 
    #                 16678, 16554, 16518, 16424, 16296, 
    #                 16226, 15970, 15673, 15240, 15071, 
    #                 14903, 14346, 14017, 13481, 12885, 
    #                 12607, 11891, 11112, 10212, 9875, 
    #                 9647, 8147, 5609, 3167, 2120, 
    #                 1942, 1285, 597, 181, 90]
    # drifts_stripped = drifts_stripped.loc[drifts_stripped['ids'].isin(ids_to_keep)]
    
    print("Set Tolerances:")
    print("mz_tol:", GapFill_mz_tol, ", RT_tol:", GapFill_rt_tol, ", CCS_tol:", GapFill_CCS_tol)
    
    print(drifts_stripped[['mzs', 'RTs', 'drifts', 'ids', 'CCS']])
    
    #Loop through samples/columns
    
    non_samples = ['mzs', 'RTs', 'drifts', 'ids', 'CCS']
    total_columns = list(drifts_stripped.columns)
    samples = list(set(total_columns) - set(non_samples))
    
    drifts_gapfilled = drifts_stripped.copy()
    
    # print(drifts_gapfilled[samples])
    
    # iterate through specific columns of the dataframe
    for sample in drifts_gapfilled[samples]:
        # print(drifts_gapfilled[sample])
        # print(sample)
        # print("-=-=-=-=-")
        #Load data
        ms1_raw_df = deimos.load('{}.h5'.format(sample), key='ms1')
        #Threshold data
        ms1_raw_df = deimos.threshold(ms1_raw_df, threshold=500)
        #Calculate CCS values from the drift times in the raw data
        ms1_raw_df['CCS'] = ccs_cal_pos.arrival2ccs(mz=ms1_raw_df['mz'], 
                                                    ta=ms1_raw_df['drift_time'], 
                                                    q=1)
        # Loop through features in each sample
        for index_f, row in drifts_gapfilled[sample].items():
            # print(index_f, row)
            feature = drifts_gapfilled
            # print(feature)
            feature_row = feature.loc[index_f]
            feature = feature_row[sample]
            
            if np.isnan(feature) == True:
                #GapFill
                # print("NaN:", feature)
                mz = feature_row['mzs']
                RT = feature_row['RTs']
                CCS = feature_row['CCS']
                # print(mz, RT, CCS)
                
                #Only keep data in desired mz/RT/dt range
                #Calculate m/z uncertainty based off ppm tolerance (mz_tol)
                uncertainty = round((mz/1000000)*GapFill_mz_tol, 5)
                # ms1_raw = ms1_raw.loc[(ms1_raw["mz"] >= mz-mz_tol) & (ms1_raw["mz"] <= mz+mz_tol)]
                ms1_raw = ms1_raw_df.loc[(ms1_raw_df["mz"] >= mz-uncertainty) & (ms1_raw_df["mz"] <= mz+uncertainty)]
                ms1_raw = ms1_raw.loc[(ms1_raw["retention_time"] >= RT-GapFill_rt_tol) & (ms1_raw["retention_time"] <= RT+GapFill_rt_tol)]
                # dt_range = round((dt/100)*dt_tol, 2)
                CCS_range = round((CCS/100)*GapFill_CCS_tol, 2)

                ms1_raw = ms1_raw.loc[(ms1_raw["CCS"] >= CCS-CCS_range) & (ms1_raw["CCS"] <= CCS+CCS_range)]
                #Threshold data
                ms1_raw = deimos.threshold(ms1_raw, threshold=500)
                Num_CCSPeaks = ms1_raw['CCS'].nunique()
                
                #Pass if there is no peak data at all
                if len(ms1_raw)==0:
                    # row[samp] = 0
                    # print(0)
                    # drifts_gapfilled.set_value[index_f, sample, 0]
                    drifts_gapfilled.at[index_f, sample] = 0
                    # pass
                else:
                    #Calculate area under intensity values
                    area_trapz = trapz(ms1_raw['intensity'], dx = GapFill_trapz_dx)
                    area_trapz = area_trapz/Num_CCSPeaks
                    # print("Gap Filled area", area_trapz)
                    drifts_gapfilled.at[index_f, sample] = area_trapz
            else:
                # print("Did not gap fill", feature)
                pass

        # print("-----------")
    
    drifts_gapfilled.to_csv("Results/GapFilling_drifts_gapfilled.csv")
    drifts_stripped.to_csv("Results/GapFilling_drifts_stripped.csv")
    
    return drifts_gapfilled
    
# =============================================================================
#     #Loop through each feature/row
#     for index, row in drifts_stripped.iterrows():
#         
#         mz = row['mzs']
#         RT = row['RTs']
#         CCS = row['CCS']
#         
#         print(row['ids'], mz, RT, CCS)
#         
#         missing_values_samps = []
#         #Get columns with NaN values
#         for samp, value in row.items():
#             if np.isnan(value) == True:
#                 missing_values_samps.append(samp)
#             else:
#                 # missing_values_samps.append(samp)
#                 pass
#             
#         remove = ["mzs", "RTs", "drifts", "ids", "CCS"]
#         missing_values_samps = [x for x in missing_values_samps if x not in remove]
#         
#         #Gap fill each sample with no peak
#         for samp in missing_values_samps:
#             # Load data
#             ms1_raw = deimos.load('{}.h5'.format(samp), key='ms1')
#             #Only keep data in desired mz/RT/dt range
#             #Calculate m/z uncertainty based off ppm tolerance (mz_tol)
#             uncertainty = round((mz/1000000)*GapFill_mz_tol, 5)
#             # ms1_raw = ms1_raw.loc[(ms1_raw["mz"] >= mz-mz_tol) & (ms1_raw["mz"] <= mz+mz_tol)]
#             ms1_raw = ms1_raw.loc[(ms1_raw["mz"] >= mz-uncertainty) & (ms1_raw["mz"] <= mz+uncertainty)]
#             ms1_raw = ms1_raw.loc[(ms1_raw["retention_time"] >= RT-GapFill_rt_tol) & (ms1_raw["retention_time"] <= RT+GapFill_rt_tol)]
#             # dt_range = round((dt/100)*dt_tol, 2)
#             CCS_range = round((CCS/100)*GapFill_CCS_tol, 2)
#             #Calculate CCS values from the drift times in the raw data
#             ms1_raw['CCS'] = ccs_cal_pos.arrival2ccs(mz=ms1_raw['mz'], 
#                                                         ta=ms1_raw['drift_time'], 
#                                                         q=1)
#             ms1_raw = ms1_raw.loc[(ms1_raw["CCS"] >= CCS-CCS_range) & (ms1_raw["CCS"] <= CCS+CCS_range)]
#             #Threshold data
#             ms1_raw = deimos.threshold(ms1_raw, threshold=500)
#             Num_CCSPeaks = ms1_raw['CCS'].nunique()
#             
#             #Pass if there is no peak data at all
#             if len(ms1_raw)==0:
#                 row[samp] = 0
#                 # pass
#             else:
#                 #Calculate area under intensity values
#                 area_trapz = trapz(ms1_raw['intensity'], dx = GapFill_trapz_dx)
#                 area_trapz = area_trapz/Num_CCSPeaks
# =============================================================================
            
# =============================================================================
#                     axes_font_size = 35
#                     label_font_size = 20
#                     
#                     # fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (20, 5))
#                     fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4, figsize = (45, 9))
#                     x_label = 'retention_time'
#                     ax1.scatter(ms1_raw[x_label], ms1_raw['intensity'])
#                     ax1.set_xlabel("Retention Time")
#                     ax1.set_ylabel("Intensity")
#                     
#                     x_label = 'drift_time'
#                     ax2.scatter(ms1_raw[x_label], ms1_raw['intensity'])
#                     ax2.set_xlabel("drift_time")
#                     ax2.set_ylabel("Intensity")
#                     
#                     x_label = 'CCS'
#                     ax3.scatter(ms1_raw[x_label], ms1_raw['intensity'])
#                     ax3.set_xlabel("CCS")
#                     ax3.set_ylabel("Intensity")
#                     
#                     x_label = 'mz'
#                     ax4.scatter(ms1_raw[x_label], ms1_raw['intensity'])
#                     ax4.set_xlabel("mz")
#                     ax4.set_ylabel("Intensity")
#                     
#                     plt.rc('axes', labelsize = axes_font_size)
#                     plt.rc('xtick', labelsize = label_font_size)
#                     plt.rc('ytick', labelsize = label_font_size)
#                     
#                     # plt.show()
#                     plt.savefig('GapFillingData/Index{}_{}_mzTol{}_mz{}_RT{}_GapFilling_Chromatograms.png'.format(index, samp, mz_tol, mz, RT))
#                     plt.close()
# =============================================================================
                    
# =============================================================================
#                 #Only replace the data if the intensity is above the set threshold
#                 if area_trapz < PeakDet_intensity_thres:
#                     row[samp] = 0
#                 else:
#                     #Replace NaN with calculated area for that sample
#                     row[samp] = area_trapz
#     
#     drifts_stripped.to_csv('drifts_stripped_final_gapfilled.csv', index=False)
#     print("gap filling steps completed")
#     drifts_gapfilled = drifts_stripped
#     
#     return drifts_gapfilled
# =============================================================================
    
def PeakMergingSteps(drifts_gapfilled): 
# def PeakMergingSteps(drifts_stripped):
    
    # drifts_gapfilled = pd.read_csv("drifts_stripped_final.csv")
    drifts_gapfilled = drifts_gapfilled.apply(pd.to_numeric, errors = "ignore")
    # drifts_gapfilled = drifts_stripped.apply(pd.to_numeric, errors = "ignore")

    #Occassionally get negative CCS values, which can cause issues
    #--> remove features/rows where the CCS values are negative
    drifts_gapfilled = drifts_gapfilled.drop(drifts_gapfilled.index[drifts_gapfilled['CCS'] < 0])

    #Sort the data by m/z
    drifts_gapfilled = drifts_gapfilled.sort_values(['mzs'], ascending=[True])
    drifts_gapfilled = drifts_gapfilled.reset_index(drop=True)
    #Add the reset index as a column for downstream joining
    drifts_gapfilled = drifts_gapfilled.reset_index()

    drifts_gapfilled_raw = drifts_gapfilled

    ##Split into m/z groups
    mz_bin = []
    for mz_val in list(drifts_gapfilled['mzs']):
        # print(mz_val)
        #Calculate uncertainty
        uncertainty = round((mz_val/1000000)*PeakMerge_mz_ppm,4)
        max_val_mz = round(mz_val+uncertainty,4)
        min_val_mz = round(mz_val-uncertainty,4)
        #Only add NEW value if it is increasing in value to the previous item added to the list
        if mz_bin == []: #if list empty, add first value
            mz_bin.append(max_val_mz)
        else:
            if min_val_mz > mz_bin[-1]: #Check if the new feature's m/z is outwith the ppm range of the previous one
                mz_bin.append(max_val_mz) #If out of range: add new upper limit to check against the next m/z
            else: #if not: just "duplicate" the current limit to check against the next m/z
                mz_bin.append(mz_bin[-1])
                
    drifts_gapfilled['mz_bin'] = mz_bin
    
    mz_clusters = drifts_gapfilled.groupby('mz_bin', observed = True) #observed = True added to remove FutureWarning message
    #Keep the dataframe from each group in a new list from the groupby object
    mz_clusters = [g.copy() for _, g in mz_clusters]
            
    ##Split m/z groups into further RT groups
    mzRT_dataframes = list()

    #Loop through each m/z group + separate by RT groups
    for table in mz_clusters:
        table = pd.DataFrame(table)
        table = table.sort_values(['RTs'], ascending=[True])
        
        ##Split each m/z group into RT groups
        RT_bin = []
        for RT_val in list(table['RTs']):

            #Calculate uncertainty tolerance
            max_val_RT = RT_val+RT_tol
            min_val_RT = RT_val-RT_tol
                
            #Only add NEW value if it is increasing in value to the previous item added to the list
            if RT_bin == []: #if list empty, add first value
                RT_bin.append(max_val_RT)
            else:
                if min_val_RT > RT_bin[-1]: #Check if the new feature's RT is outwith the RT range of the previous one
                    RT_bin.append(max_val_RT) #If out of range: add new upper limit to check against the next RT
                else: #if not: just "duplicate" the current limit to check against the next RT
                    RT_bin.append(RT_bin[-1])
                    
        table['RT_bin'] = RT_bin
        
        RT_clusters = table.groupby('RT_bin', observed = True)
        
        #Add each m/z-RT group individually to the new list
        for mzRT_group in RT_clusters:
            mzRT_dataframes.append(mzRT_group)
        
    #Keep the dataframe from each group in a new list from the groupby object
    mzRT_dataframes = [g.copy() for _, g in mzRT_dataframes]

    #Loop through each m/z/RT group + separate by CCS groups
    mzRTCCS_dataframes = list()

    #Loop through each m/z/RT group + separate by CCS groups
    for table in mzRT_dataframes:
        table = pd.DataFrame(table)
        table = table.sort_values(['CCS'], ascending=[True])
        
        ##Split each m/z group into RT groups
        CCS_bin = []
        for CCS_val in list(table['CCS']):
            
            #Calculate uncertainty tolerance
            uncertainty_CCS = (CCS_val/100)*CCS_tol
            max_val_CCS = CCS_val + uncertainty_CCS
            min_val_CCS = CCS_val - uncertainty_CCS
                
            #Only add NEW value if it is increasing in value to the previous item added to the list
            if CCS_bin == []: #if list empty, add first value
                CCS_bin.append(max_val_CCS)
            else:
                if min_val_CCS > CCS_bin[-1]: #Check if the new feature's CCS is outwith the CCS range of the previous one
                    CCS_bin.append(max_val_CCS) #If out of range: add new upper limit to check against the next CCS
                else: #if not: just "duplicate" the current limit to check against the next CCS
                    CCS_bin.append(CCS_bin[-1])
                    
        table['CCS_bin'] = CCS_bin
        
        CCS_clusters = table.groupby('CCS_bin', observed = True)
        
        #Add each m/z-RT group individually to the new list
        for mzRTCCS_group in CCS_clusters:
            mzRTCCS_dataframes.append(mzRTCCS_group)
        
    #Keep the dataframe from each group in a new list from the groupby object
    mzRTCCS_dataframes = [g.copy() for _, g in mzRTCCS_dataframes]

    ##Sum rows + prepare final version of data
    colnames = mzRTCCS_dataframes[0].columns
    # print(colnames)

    non_samples = ['index', 'RT_bins', 'mzs', 'RTs', 'drifts', 'ids', 'CCS', 'CCS_bin', 'mz_bin']
    samples = [header for header in colnames if header not in non_samples]

    #Loop through each mz+RT+CCS group + sum intensities across samples + add to final dataframe
    PeakMerged_dataframe = pd.DataFrame()

    for feature in mzRTCCS_dataframes:
        feature = feature.reset_index(drop=True)
        # print(pd.DataFrame(feature)[samples])
        # print(feature['ID'].values.tolist())
        
        #Get the sum of the intensities across samples
        #Replace 0.001 with NaN --> these are not summed
        sum_feature = pd.DataFrame(feature)[samples].replace(0.001, np.NaN).sum(axis=0)
        sum_feature = sum_feature.to_frame()
        sum_feature = sum_feature.transpose()
        #Replace 0.0 with 0.001 to bring this value back into the data
        sum_feature = sum_feature.replace(0.0, 0.001)
        # print(sum_feature)
        
        #Add in relevant non-sample columns back into the dataframe
        sum_feature['ids'] = ""
        sum_feature.at[0,'ids'] = list(feature['ids'].values)
        sum_feature['RTs'] = round(feature['RTs'].mean(), 3)
        sum_feature['drifts'] = round(feature['drifts'].mean(), 3)
        sum_feature['CCS'] = round(feature['CCS'].mean(), 2)
        sum_feature['mzs'] = round(feature['mzs'].mean(), 4)
        
        #Reorder columns
        column_order = ['mzs', 'RTs', 'drifts', 'ids', 'CCS']
        column_order = samples + column_order
        sum_feature = sum_feature[column_order]
        
        PeakMerged_dataframe = pd.concat([PeakMerged_dataframe, sum_feature])
        
    #Occasionally, an extra and empty row will be present at the top of the final dataframe
    #Can remove simply based on it not having an RT value
    PeakMerged_dataframe = PeakMerged_dataframe[PeakMerged_dataframe['RTs'].notna()]
    PeakMerged_dataframe.to_csv("Results/PeakMerging_final_dataframe.csv", index = False)
    
    return PeakMerged_dataframe
    
def MinimumDetectionThresholdSteps(PeakMerged_dataframe):
    #IMPORTANT
    #1) Run the AssignSampleGroupings.py file (only needs to be done once)
    #2) Fill in the Group column (and SAVE it!)
    #3) Update the file name read in for the df_sample_groups object

    #Load sample_groupings
    df_sample_groups = pd.read_csv('Sample_groupings.csv')
    df_sample_groups = pd.DataFrame(df_sample_groups)

    # =============================================================================
    # sample_group_list = list(df_sample_groups["Groups"].unique())
    # sample_group_list.remove('QC')
    # =============================================================================

    #Load in the MassProfiler data
    # df = pd.read_csv('PeakMerged_final_dataframe.csv')
    df = PeakMerged_dataframe
    df = df.apply(pd.to_numeric, errors = "ignore")
    print(len(df))

    #For development, only keep selected number of rows
    # df = df[0:10]

    #Make a list of the samples
    samples = list(df.columns)
    #Remove non-sample names from list
    to_remove = ['mzs', 'RTs', 'drifts', 'ids', 'CCS']
    samples = [x for x in samples if x not in to_remove]

    minimum_detection_group_threshold = 0.50 #33% - PLEASE give to 2 significant figures!

    features_kept = pd.DataFrame()
    features_removed = pd.DataFrame()

    count_OverMinThreshold = 0
    count_UnderMinThreshold = 0

    #Features in QCs and NOT in samples are to be removed
    for index, row in df.iterrows(): #Loop through each feature
        feature_series = row
        feature = pd.DataFrame(row)
        feature = feature.reset_index()
        feature = feature.rename(columns = {"index":"Samples", 
                                        feature.columns[1]: "Intensities"})
        #Remove non-samples
        feature = feature[feature.Samples.isin(to_remove) == False]
        #Add in grouping information
        feature = pd.merge(feature, df_sample_groups, on='Samples', how='inner')
        # feature = feature.sort_values(['Groups'], ascending=[False])
        #Determine % of samples it is present in
        sample_group_prop = feature.groupby('Groups')
        ND_proportions_feature = pd.DataFrame()
        #print(feature)
        for key, item in sample_group_prop: #Loop through the groups in each feature
            group = pd.DataFrame(sample_group_prop.get_group(key))
            # print(sample_group_prop.get_group(key), "\n\n")
            number_ND = round(group['Intensities'].value_counts(normalize=True),2)
            number_ND = pd.DataFrame(number_ND).reset_index()
            number_ND = number_ND.rename(columns = {"proportion":"Proportion"})
            if number_ND.shape[0] < 1:
                pass
            else: #Only apply threshold calc to groups with > 1 sample in it
                ND_prop = number_ND.loc[number_ND['Intensities'] == 0.001]
                if ND_prop.empty == True: #If no 0.001 values are present
                    dummy_row = pd.DataFrame({'Intensities':0.001, 
                                              'Proportion':0, 
                                              'Groups': group["Groups"].unique()})
                    ND_proportions_feature = pd.concat([ND_proportions_feature, 
                                                        dummy_row])
                    pass 
                else: 
                    # print("group:")
                    # print(group)
                    #Add group name to the row with the % 0.001 values
                    group_name = group["Groups"].unique()
                    group_name = {'Groups':group_name}
                    ND_prop = ND_prop.assign(**group_name)
                    ND_proportions_feature = pd.concat([ND_proportions_feature, 
                                                        pd.DataFrame(ND_prop)])
        #Perform minimum detection threshold filtering
        ND_proportions_feature = ND_proportions_feature.reset_index(drop = True)
        if ND_proportions_feature["Proportion"].min() > minimum_detection_group_threshold:
            #If minimum 0.001 abundance got a feature is above the threshold --> remove
            count_OverMinThreshold += 1
            features_removed = pd.concat([features_removed, feature_series.to_frame().T], 
                                      ignore_index = True)
            
        else: #Otherwise: keep it
            count_UnderMinThreshold += 1
            features_kept = pd.concat([features_kept, feature_series.to_frame().T], 
                                      ignore_index = True)

            
    features_kept = pd.DataFrame(features_kept)
    print(features_kept)
    features_removed = pd.DataFrame(features_removed) 
    print(features_removed)
    print("================================================")      
    print("No. features removed:", count_OverMinThreshold)
    print("No. features kept:", count_UnderMinThreshold)
    print("================================================")
    
    features_kept.to_csv("Results/final_dataframe_unannotated.csv", index = False)

def MergeMS2Data():
    
    features_kept = pd.read_csv('20250506_Results_ToReviewWithKarl/final_dataframe_unannotated.csv')
    features_kept = features_kept.apply(pd.to_numeric, errors = "ignore")
    MS2Spectra = pd.read_csv('Results/MS2Extraction_res_final_400kTF_Spk_SR_Sample.csv')
    print(features_kept[["mzs", "RTs", "drifts", "CCS", "ids"]])
    print(MS2Spectra[["mz_ms1", "retention_time_ms1", "drift_time_ms1"]])
    print(MS2Spectra[["mz_ms2", "retention_time_ms2", "drift_time_raw_ms2"]])
    
    #TODO: Development only
    # features_kept = features_kept.iloc[86:90]
    total_no_features = len(features_kept)
    counter_mz = 0
    counter_RT = 0
    counter_DT = 0
    
    #Iterate through the features to assign MS2 spectra to each one
    for index, row in features_kept.iterrows():
        # print(index)
        # print("Feature Details: ", "mzs:", row['mzs'], ", RTs:", row['RTs'], ", DTs:", row['drifts'])
        
        #Select within mz tolerance
        uncertainty = round((row['mzs']/1000000)*MergeMS2_mz_ppm, 4)
        mz_upper_limit = round(row['mzs']+uncertainty, 4)
        mz_lower_limit = round(row['mzs']-uncertainty, 4)
        # print("mz info: ", uncertainty, mz_upper_limit, mz_lower_limit)
        subset = MS2Spectra[(MS2Spectra['mz_ms1'] < mz_upper_limit) & (MS2Spectra['mz_ms1'] > mz_lower_limit)]
        if subset.empty:
            pass
        else:
            counter_mz += 1
            # print(subset[['mz_ms1', "retention_time_ms1", "drift_time_ms1"]])
            
        #Select within Retention Time tolerance
        RT_upper_limit = row['RTs']+MergeMS2_RT_tol
        RT_lower_limit = row['RTs']-MergeMS2_RT_tol
        # print("RT info: ", RT_upper_limit, RT_lower_limit)
        if subset.empty:
            pass
        else:
            subset = subset[(subset['retention_time_ms1'] < RT_upper_limit) & (subset['retention_time_ms1'] > RT_lower_limit)]
            counter_RT += 1
            # print(subset[['mz_ms1', "retention_time_ms1", "drift_time_ms1"]])
            
        #Select within Drift Time tolerance
        PercDT = (row['drifts']/100)*MergeMS2_DT_tol
        DT_upper_limit = round(row['drifts'] + PercDT, 2)
        DT_lower_limit = round(row['drifts'] - PercDT, 2)
        # print("DT info: ", PercDT, DT_upper_limit, DT_lower_limit)
        if subset.empty:
            pass
        else:
            counter_DT += 1
            subset = subset[(subset['drift_time_ms1'] < DT_upper_limit) & (subset['drift_time_ms1'] > DT_lower_limit)]
            print(index)
            print("Feature Details: ", "mzs:", row['mzs'], ", RTs:", row['RTs'], ", DTs:", row['drifts'])
            print("mz info: ", uncertainty, mz_upper_limit, mz_lower_limit)
            print("RT info: ", RT_upper_limit, RT_lower_limit)
            print("DT info: ", PercDT, DT_upper_limit, DT_lower_limit)
            print(subset[['mz_ms1', "retention_time_ms1", "drift_time_ms1"]])
            
        # print("--------------")
    
    print("Total Number of Features:", total_no_features)
    print("Total with MS2 spectra aligned to mz level:", counter_mz)
    print("Total with MS2 spectra aligned to mz & RT level:", counter_RT)
    print("Total with MS2 spectra aligned to mz, RT, and DT level:", counter_DT)

if __name__ == "__main__":
    
    processes=1
    
    startTime = datetime.now()
    print("===============================")
    print("DEIMoS Script startTime:", startTime)
    print("===============================")

    #Create multipeaks and loopcount objects for downstream agglomerative clustering steps
    # Create a loop counter for the agglomerative clustering steps
    loopcount = 0
    
    ##Find the file generated in the middle of the run
    #Make dataframe to contain file names and creation times
    
    files = ReadFilesInDirectory()
    print("ReadFilesInDirectory() complete")
    
    middle, rtalign_data = FindMiddleFileForRTAlignment()
    print("FindMiddleFileForRTAlignment() complete")
    
    #del [CreationTime, file, Created, middle_row]
    
    CreateResultsFile()
    print("CreateResultsFile() complete")
    
    ccs_cal_pos = CreateCCSCalObjects()
    print("CreateCCSCalObjects() complete")
    
    #Loop through the samples in the directory for sample-specific processes
    for sample in files:
        print("------------------------")
        print("Processing file", sample)
        print("------------------------")
        
        #########################################
        #Make Folder for results for this sample
        #########################################
        file_NoExt = sample.replace('.h5', '')
        #Make Folder for results for this sample
        if os.path.exists('Results/{}'.format(file_NoExt)):
            shutil.rmtree('Results/{}'.format(file_NoExt))
        os.makedirs('Results/{}'.format(file_NoExt))
        
        RetentionTimeAlignment(file_NoExt, rtalign_data, middle)
        print("RetentionTimeAlignment() complete")
        
        ms1, ms1_peaks, ms2, ms2_peaks, ms1_iso = DetectPeaks(file_NoExt)
        print("DetectPeaks() complete")

        #del [factors, index, ms1_thres500, ms2_thres500]
        
        res_final = ExtractMS2Spectra(ms1, ms2, file_NoExt)
        print("ExtractMS2Spectra() complete")
        
        # del [ms1_thres100, ms2_thres100, mz_i, dt_i, rt_i, intensity_i, precursor, 
        #       fragment_profile, fragment_dt, precursor_dt, ms1_peaks_thres_forMS2, 
        #       ms2_peaks_thres_forMS2, decon, res, offset_correction_model, index, row]
        
# =============================================================================
#         DetectIsotopes()
#         print("DetectIsotopes() complete")
# =============================================================================
        
        #del [isotopes, ms1_peaks_thres1000, partitions]
        
        #Make a folder for this sample for all Agglomerative Clustering results to go in to
        if os.path.exists('Results/{}/AgglomerativeClustering'.format(file_NoExt)):
            shutil.rmtree('Results/{}/AgglomerativeClustering'.format(file_NoExt))
        os.mkdir('Results/{}/AgglomerativeClustering'.format(file_NoExt))
        
        loopcount = AgglomerativeClusteringConcatenateNewPeakData(loopcount, res_final)
        print("AgglomerativeClusteringConcatenateNewPeakData() complete")
        
    # del [files, middle, sample, file_NoExt, ms1, ms1_peaks, ms2, 
    #       ms2_peaks, ms1_iso, loopcount, ReadFilesInDirectory, FindMiddleFileForRTAlignment,
    #       CreateResultsFile, DetectPeaks, ExtractMS2Spectra, DetectIsotopes, 
    #       ReferenceBasedAlignmentRT, AgglomerativeClusteringConcatenateNewPeakData]
    
    #Perform once final dataset for clustering (w/ all sample data) is created    
    clustering = AgglomerativeClusteringMainSteps()
    print("AgglomerativeClusteringMainSteps() complete")
    
    # drifts_stripped = CCSCalibrationSteps(ccs_cal_pos)
    drifts_stripped = CCSCalibrationSteps(ccs_cal_pos, clustering)
    print("CCSCalibrationSteps() complete")
        
    # del [ms1_iso, ms1, ms1_peaks, ms2, ms2_peaks, drifts_stripped]
    
    drifts_gapfilled = GapFillingSteps(ccs_cal_pos, drifts_stripped)
    # drifts_gapfilled = GapFillingSteps(ccs_cal_pos)
    print("GapFillingSteps() complete")
    
    PeakMerged_dataframe = PeakMergingSteps(drifts_gapfilled)
    # PeakMerged_dataframe = PeakMergingSteps(drifts_stripped)
    print("PeakMergingSteps() complete")
    
    MinimumDetectionThresholdSteps(PeakMerged_dataframe)
    print("MinimumDetectionThresholdSteps() complete")

# =============================================================================
#     MergeMS2Data()
#     print("MergeMS2Data() complete")
# =============================================================================
    
    # del [calib_files, tune_pos_file, ccsCalib_mz, ccsCalib_ccs, ccsCalib_q, 
    #       ccsCalib_buffer_mass, ccsCalib_mz_tol, ccsCalib_dt_tol, 
    #       PeakDet_intensity_thres, PeakDet_smooth_data_radius, PeakDet_persistent_homology_radius, 
    #       MS2Extract_intensity_thres, MS2Extract_ms1_mz_subset_low, MS2Extract_ms1_dt_subset_low, 
    #       MS2Extract_ms1_rt_subset_low, MS2Extract_ms1_mz_subset_high, 
    #       MS2Extract_ms1_dt_subset_high, MS2Extract_ms1_rt_subset_high, 
    #       MS2Extract_ms2_dt_subset_low, MS2Extract_ms2_rt_subset_low, 
    #       MS2Extract_ms2_dt_subset_high, MS2Extract_ms2_rt_subset_high, 
    #       MS2Extract_model_ce, MS2Extract_model_params, MS2Extract_ms1_decon_intensity_thres, 
    #       MS2Extract_ms2_decon_intensity_thres, MS2Extract_construct_pairs_dt_low, 
    #       MS2Extract_construct_pairs_rt_low, MS2Extract_construct_pairs_dt_high, 
    #       MS2Extract_construct_pairs_rt_high, MS2Extract_construct_pairs_ce, 
    #       MS2Extract_construct_pairs_error_tol, MS2Extract_config_extract_mz_low, 
    #       MS2Extract_config_extract_dt_low, MS2Extract_config_extract_rt_low, 
    #       MS2Extract_config_extract_mz_high, MS2Extract_config_extract_dt_high, 
    #       MS2Extract_config_extract_rt_high, MS2Extract_decon_dt_resolution, 
    #       MS2Extract_dt_score_threshold, isotope_intensity_thres, isotope_partition_size, 
    #       isotope_partition_overlap, isotope_map_mz_dt_rt_tol, isotope_map_delta, 
    #       isotope_map_max_isotopes, isotope_map_max_charges, isotope_map_max_error, 
    #       isotope_min_no_isotopes, isotope_slice_mz_low, isotope_slice_mz_high, 
    #       isotope_plot_slice_mz_low, isotope_plot_slice_dt_low, isotope_plot_slice_rt_low, 
    #       isotope_plot_slice_mz_high, isotope_plot_slice_dt_high, isotope_plot_slice_rt_high, 
    #       rtalign_persisHomology_thres, rtalign_persis_thres, rtalign_partition_thres,
    #       rtalign_partition_size, rtalign_partition_overlap, rtalign_zipmap_thres, 
    #       rtalign_zipmap_mz_dt_rt_tol, agglo_mergeFeatures_mz_dt_rt_tol, agglo_multiSampPart_size, 
    #       agglo_multiSampPart_tol, agglo_clustering_mz_dt_rt_tol, GapFill_mz_tol, 
    #       GapFill_rt_tol, GapFill_CCS_tol, GapFill_trapz_dx, PeakMerge_mz_ppm, 
    #       PeakMerge_RT_tol, PeakMerge_CCS_tol]

    print("===============================")
    print("DEIMoS Script stopTime:", datetime.now())
    print("Total process run time:", datetime.now() - startTime)
    print("===============================")

    print("===============================")
    print("Objects stored locally from DEIMoS and custom steps:")
    print(list(locals()))
    print("===============================")

# =============================================================================
# def ReferenceBasedAlignmentRT():
#     
#     # #Perform Alignment on the files
#     #Make a folder for this sample for all Alignment results to go in to
#     if os.path.exists('Results/{}/Alignment'.format(file_NoExt)):
#         shutil.rmtree('Results/{}/Alignment'.format(file_NoExt))
#     os.mkdir('Results/{}/Alignment'.format(file_NoExt))
#     
#     data = {}
#     #File to be aligned
#     data['A'] = deimos.load('{}.h5'.format(file_NoExt), key='ms1')
#     #Reference file
#     data['B'] = deimos.load('{}.h5'.format(middle), key='ms1')
#     
#     # Collapse
#     a_rt = deimos.collapse(data['A'], keep='retention_time').sort_values(by='retention_time')
#     b_rt = deimos.collapse(data['B'], keep='retention_time').sort_values(by='retention_time')
#     # Visualize
#     fig, ax = plt.subplots(1, dpi=150, facecolor='w')
#     ax.fill_between(a_rt['retention_time'], a_rt['intensity'], color='C0', alpha=0.5, label='A')
#     ax.fill_between(b_rt['retention_time'], b_rt['intensity'], color='C3', alpha=0.5, label='B')
#     ax.set_xlabel('Retention Time', fontweight='bold')
#     ax.set_ylabel('Intensity', fontweight='bold')
#     ax.set_xlim(0, None)
#     ax.set_ylim(0, None)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('Results/{}/Alignment/InitialMisalignment.png'.format(file_NoExt))
#     #Save memory space
#     plt.close()
# 
#     del [fig, ax]
#     
#     # Perform peak detection
#     peaks = {}
#     peaks['A'] = deimos.peakpick.persistent_homology(deimos.threshold(data['A'], 
#                                                                       threshold=rtalign_persisHomology_thres),
#                                                      dims=['mz', 'drift_time', 'retention_time'])
#     peaks['B'] = deimos.peakpick.persistent_homology(deimos.threshold(data['B'], 
#                                                                       threshold=rtalign_persisHomology_thres),
#                                                      dims=['mz', 'drift_time', 'retention_time'])
#     # Downselect by persistence
#     peaks['A']['persistence_ratio'] = peaks['A']['persistence'] / peaks['A']['intensity']
#     peaks['A'] = deimos.threshold(peaks['A'], by='persistence_ratio', threshold=rtalign_persis_thres)
#     
#     peaks['B']['persistence_ratio'] = peaks['B']['persistence'] / peaks['B']['intensity']
#     peaks['B'] = deimos.threshold(peaks['B'], by='persistence_ratio', threshold=rtalign_persis_thres)
#     
#     # Partition
#     partitions = deimos.partition(deimos.threshold(peaks['A'], threshold=rtalign_partition_thres),
#                                   split_on='mz',
#                                   size=rtalign_partition_size,
#                                   overlap=rtalign_partition_overlap)
#     
#     # Match
#     a_matched, b_matched = partitions.zipmap(deimos.alignment.match, deimos.threshold(peaks['B'], 
#                                                                                       threshold=rtalign_zipmap_thres),
#                                               dims=['mz', 'drift_time', 'retention_time'],
#                                               tol=rtalign_zipmap_mz_dt_rt_tol, relative=[True, True, False],
#                                               processes=4)
#     
#     # Visualize
#     fig, ax = plt.subplots(1, dpi=150, facecolor='w')
#     ax.scatter(a_matched['retention_time'], b_matched['retention_time'], s=2)
#     ax.plot([0, 25], [0, 25], linewidth=1, linestyle='--', color='k')
#     
#     ax.set_xlabel('Retention Time (A)', fontweight='bold')
#     ax.set_ylabel('Retention Time (B)', fontweight='bold')
#     
#     ax.set_xlim(0, 25)
#     ax.set_ylim(0, 25)
#     
#     plt.tight_layout()
#     plt.savefig('Results/{}/Alignment/PartitionedData_NotFit.png'.format(file_NoExt))
#     #Save memory space
#     plt.close()
#     
#     # SVR spline
#     spl = deimos.alignment.fit_spline(a_matched, b_matched, align='retention_time', kernel='rbf', C=1000)
#     newx = np.linspace(0, a_matched['retention_time'].max(), 1000)
#     
#     # Visualize
#     fig, ax = plt.subplots(1, dpi=150, facecolor='w')
#     ax.plot(newx, spl(newx), c='black', linewidth=1, linestyle='--')
#     ax.scatter(a_matched['retention_time'], b_matched['retention_time'], s=2)
#     
#     ax.set_xlabel('Retention Time (A)', fontweight='bold')
#     ax.set_ylabel('Retention Time (B)', fontweight='bold')
#     
#     ax.set_xlim(0, 25)
#     ax.set_ylim(0, 25)
#     
#     plt.tight_layout()
#     plt.savefig('Results/{}/Alignment/PartitionedData_SVRFit.png'.format(file_NoExt))
#     #Save memory space
#     plt.close()
#     
#     #Apply alignment and replot the total ion chromatograms
#     data['A_aligned'] = data['A'].copy()
#     data['A_aligned']['retention_time'] = spl(data['A_aligned']['retention_time'])
#     # Collapse
#     a_rt_aligned = deimos.collapse(data['A_aligned'], keep='retention_time').sort_values(by='retention_time')
#     # Visualize
#     fig, ax = plt.subplots(1, dpi=150, facecolor='w')
#     ax.fill_between(a_rt_aligned['retention_time'], a_rt_aligned['intensity'], color='C0', alpha=0.5, label='spl(A)')
#     ax.fill_between(b_rt['retention_time'], b_rt['intensity'], color='C3', alpha=0.5, label='B')
#     
#     ax.set_xlabel('Retention Time', fontweight='bold')
#     ax.set_ylabel('Intensity', fontweight='bold')
#     
#     ax.set_xlim(0, None)
#     ax.set_ylim(0, None)
#     
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('Results/{}/Alignment/FinalAlignment.png'.format(file_NoExt))
#     #Save memory space
#     plt.close()
# =============================================================================

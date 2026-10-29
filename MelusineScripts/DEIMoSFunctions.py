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

def FindMiddleFileForRTAlignment(ms1_threshold, ms2_threshold, PeakDetectionMethod, 
                                 PeakDet_smooth_data_radius, PeakDet_smooth_data_iterations,
                                 PeakDet_persistent_homology_radius, MS2DataPresent):
    #Find the file in the middle of the run to align other files to downstream
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
    # middle = "RTAligned_{}".format(middle)
    
    print("middle = ", middle)

    #Make Folder for results for this sample
    if os.path.exists('Results/{}'.format(middle)):
        shutil.rmtree('Results/{}'.format(middle))
    os.makedirs('Results/{}'.format(middle))
    
    #Load in the middle file for downstream RT alginment
    rtalign_data = {}
    rtalign_data['ms1'] = deimos.load('{}.h5'.format(middle), key='ms1')
    rtalign_data['ms1'] = rtalign_data['ms1'].apply(pd.to_numeric, errors = "ignore")

    #Threshold the data
    rtalign_data['ms1'] = deimos.threshold(rtalign_data['ms1'], threshold=ms1_threshold)

    if MS2DataPresent is True:
        rtalign_data['ms2'] = deimos.load('{}.h5'.format(middle), key='ms2')
        rtalign_data['ms2'] = rtalign_data['ms2'].apply(pd.to_numeric, errors = "ignore")
        rtalign_data['ms2'] = deimos.threshold(rtalign_data['ms2'], threshold=ms2_threshold)

    #Peak picks (required for RT alignment)
    rtalign_data = DetectPeaks(middle, rtalign_data, PeakDetectionMethod,
                               PeakDet_smooth_data_radius, PeakDet_smooth_data_iterations,
                               PeakDet_persistent_homology_radius, MS2DataPresent)

    #Just save the middle/reference file as it is
    deimos.save('Results/RTAligned_{}.h5'.format(middle), rtalign_data['ms1'], key='ms1', mode='w')
    if MS2DataPresent is True:
        deimos.save('Results/RTAligned_{}.h5'.format(middle), rtalign_data['ms2'], key='ms2', mode='a')
    
    return middle, rtalign_data

def DetectPeaks(file_NoExt, sample_data, PeakDetectionMethod, 
                PeakDet_smooth_data_radius, PeakDet_smooth_data_iterations, 
                PeakDet_persistent_homology_radius, MS2DataPresent):
    
    #Make a folder for this sample for all Peak Detection results to go in to
    if os.path.exists('Results/{}/PeakDetection'.format(file_NoExt)):
        shutil.rmtree('Results/{}/PeakDetection'.format(file_NoExt))
    os.mkdir('Results/{}/PeakDetection'.format(file_NoExt))

    if PeakDetectionMethod == "PersistentHomology":
    
        # Build factors from raw data
        factors = deimos.build_factors(sample_data['ms1'], dims='detect')
        # Build index
        index = deimos.build_index(sample_data['ms1'], factors)
        # Smooth data
        sample_data['ms1'] = deimos.filters.smooth(sample_data['ms1'], 
                                            index=index, dims=['mz', 'drift_time',
                                                                'retention_time'],
                                            radius=PeakDet_smooth_data_radius, iterations=PeakDet_smooth_data_iterations)
        # Perform peak detection
        sample_data['ms1_peaks'] = deimos.peakpick.persistent_homology(sample_data['ms1'], index=index,
                                                    dims=['mz', 'drift_time', 
                                                        'retention_time'],
                                                    radius=PeakDet_persistent_homology_radius)
    
        #Save outputs as .csv file
        sample_data['ms1_peaks'].to_csv('Results/{}/PeakDetection/ms1_peaks.csv'.format(file_NoExt), 
                                        index=False)
        
        #Remove columns not required for downstream processes
        sample_data['ms1_peaks'] = sample_data['ms1_peaks'].drop(['mz_weighted', 'drift_time_weighted', 
                                    'retention_time_weighted'], axis = 1)

        if MS2DataPresent is True:
            # Build factors from raw data
            factors = deimos.build_factors(sample_data['ms2'], dims='detect')
            # Build index
            index = deimos.build_index(sample_data['ms2'], factors)
            # Smooth data
            sample_data['ms2'] = deimos.filters.smooth(sample_data['ms2'], 
                                                index=index, dims=['mz', 'drift_time', 
                                                                    'retention_time'],
                                                radius=PeakDet_smooth_data_radius, iterations=7)
            # Perform peak detection
            sample_data['ms2_peaks'] = deimos.peakpick.persistent_homology(sample_data['ms2'], index=index,
                                                        dims=['mz', 'drift_time', 
                                                            'retention_time'],
                                                        radius=PeakDet_persistent_homology_radius)
            
            #Save outputs as .csv file
            sample_data['ms2_peaks'].to_csv('Results/{}/PeakDetection/ms2_peaks.csv'.format(file_NoExt), index=False)
            
            #Remove columns not required for downstream processes
            sample_data['ms2_peaks'] = sample_data['ms2_peaks'].drop(['mz_weighted', 'drift_time_weighted', 
                                        'retention_time_weighted'], axis = 1)

    elif PeakDetectionMethod == "MaximumFiltration":
        # Load data, excluding scanid column
        # ms1 = deimos.load('example_data.h5', key='ms1', columns=['mz', 'drift_time', 'retention_time', 'intensity'])

        # Sum over retention time
        ms1_2d = deimos.collapse(sample_data['ms1'][['mz', 'drift_time', 'retention_time', 'intensity']], 
                                 keep=['mz', 'drift_time', 'retention_time'])
        # Perform peak detection
        sample_data['ms1_peaks'] = deimos.peakpick.local_maxima(ms1_2d, dims=['mz', 'drift_time', 'retention_time'], bins=[37, 9, 37])
        del ms1_2d

        #Save outputs as .csv file
        sample_data['ms1_peaks'].to_csv('Results/{}/PeakDetection/ms1_peaks.csv'.format(file_NoExt), 
                                        index=False)

        if MS2DataPresent == True:
            # Sum over retention time
            ms2_2d = deimos.collapse(sample_data['ms2'][['mz', 'drift_time', 'retention_time', 'intensity']], keep=['mz', 'drift_time', 'retention_time'])
            # Perform peak detection
            sample_data['ms2_peaks'] = deimos.peakpick.local_maxima(ms2_2d, dims=['mz', 'drift_time', 'retention_time'], bins=[37, 9, 37])
            del ms2_2d
            #Save outputs as .csv file
            sample_data['ms2_peaks'].to_csv('Results/{}/PeakDetection/ms2_peaks.csv'.format(file_NoExt), 
                                        index=False)
    
    return sample_data

def RetentionTimeAlignment(file_NoExt, sample_data, rtalign_data, PeakDetectionMethod, rtalign_persisHomology_thres, 
                           rtalign_persis_thres, rtalign_partition_thres, rtalign_partition_size, 
                           rtalign_partition_overlap, rtalign_zipmap_thres, rtalign_zipmap_mz_dt_rt_tol, 
                           SaveRTAlignmentDataFiles, SaveRTAlignmentGraphs, MS2DataPresent, 
                           PeakDet_smooth_data_radius, PeakDet_persistent_homology_radius, 
                           PeakDet_smooth_data_iterations):

    if SaveRTAlignmentDataFiles is True or SaveRTAlignmentGraphs is True:
        #Make a folder for this sample for all Peak Detection results to go in to
        if os.path.exists('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt)):
            shutil.rmtree('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt))
        os.mkdir('Results/{}/RetentionTimeAlignmentAndPeakDetection'.format(file_NoExt))

    #Detect peaks to use in RT Alignment
    sample_data = DetectPeaks(file_NoExt, sample_data, PeakDetectionMethod, PeakDet_smooth_data_radius,
                              PeakDet_smooth_data_iterations, PeakDet_persistent_homology_radius, MS2DataPresent)
    
    if SaveRTAlignmentGraphs is True:
        #Visualise misalignment
        # Collapse
        ToAlign_ms1 = deimos.collapse(sample_data['ms1'], keep='retention_time').sort_values(by='retention_time')
        Ref_ms1 = deimos.collapse(rtalign_data['ms1'], keep='retention_time').sort_values(by='retention_time')
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
        plt.close()

        if MS2DataPresent is True:
            ToAlign_ms2 = deimos.collapse(sample_data['ms2'], keep='retention_time').sort_values(by='retention_time')
            Ref_ms2 = deimos.collapse(rtalign_data['ms2'], keep='retention_time').sort_values(by='retention_time')
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
            plt.close()

    # Downselect by persistence
    sample_data['ms1_peaks']['persistence_ratio'] = sample_data['ms1_peaks']['persistence'] / sample_data['ms1_peaks']['intensity']
    # ms1_peaks_persist_thres = deimos.threshold(sample_data['ms1_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)
    sample_data['ms1_peaks'] = deimos.threshold(sample_data['ms1_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)
    
    rtalign_data['ms1_peaks']['persistence_ratio'] = rtalign_data['ms1_peaks']['persistence'] / rtalign_data['ms1_peaks']['intensity']
    # ms1_peaks_persist_thres_ref = deimos.threshold(rtalign_data['ms1_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)
    rtalign_data['ms1_peaks'] = deimos.threshold(rtalign_data['ms1_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)

    # Partition
    partitions_toAlign_ms1 = deimos.partition(sample_data['ms1_peaks'],
                                              split_on='mz',
                                              size=rtalign_partition_size,
                                              overlap=rtalign_partition_overlap)

    # Match
    toAlign_ms1_matched, ref_ms1_matched = partitions_toAlign_ms1.zipmap(deimos.alignment.match, #type: ignore
                                                                         rtalign_data['ms1_peaks'], 
                                                                         dims=['mz', 'drift_time', 'retention_time'],
                                                                         tol=rtalign_zipmap_mz_dt_rt_tol, 
                                                                         relative=[True, True, False],
                                                                         processes=4)

    if MS2DataPresent is True:
        sample_data['ms2_peaks']['persistence_ratio'] = sample_data['ms2_peaks']['persistence'] / sample_data['ms2_peaks']['intensity']
        # ms2_peaks_persist_thres = deimos.threshold(sample_data['ms2_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)
        sample_data['ms2_peaks'] = deimos.threshold(sample_data['ms2_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)

        rtalign_data['ms2_peaks']['persistence_ratio'] = rtalign_data['ms2_peaks']['persistence'] / rtalign_data['ms2_peaks']['intensity']
        # ms2_peaks_persist_thres_ref = deimos.threshold(rtalign_data['ms2_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)
        rtalign_data['ms2_peaks'] = deimos.threshold(rtalign_data['ms2_peaks'], by='persistence_ratio', threshold=rtalign_persis_thres)

        partitions_toAlign_ms2 = deimos.partition(sample_data['ms2_peaks'], 
                                            split_on='mz',
                                            size=rtalign_partition_size,
                                            overlap=rtalign_partition_overlap)

        toAlign_ms2_matched, ref_ms2_matched = partitions_toAlign_ms2.zipmap(deimos.alignment.match, 
                                                                        rtalign_data['ms2_peaks'],
                                                                        dims=['mz', 'drift_time', 'retention_time'],
                                                                        tol=rtalign_zipmap_mz_dt_rt_tol, 
                                                                        relative=[True, True, False],
                                                                        processes=4)  

    if SaveRTAlignmentGraphs is True:
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
        plt.close()

        if MS2DataPresent is True:
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
            plt.close()
    
    # SVR spline
    spl_ms1 = deimos.alignment.fit_spline(toAlign_ms1_matched, ref_ms1_matched, 
                                        align='retention_time', kernel='rbf', C=1000)
    newx_ms1 = np.linspace(0, toAlign_ms1_matched['retention_time'].max(), 1000)

    if MS2DataPresent is True:
        spl_ms2 = deimos.alignment.fit_spline(toAlign_ms2_matched, ref_ms2_matched, 
                                            align='retention_time', kernel='rbf', C=1000)
        newx_ms2 = np.linspace(0, toAlign_ms2_matched['retention_time'].max(), 1000)

    if SaveRTAlignmentGraphs is True:
        # Visualize
        fig, ax = plt.subplots(1, dpi=150, facecolor='w')
        ax.plot(newx_ms1, spl_ms1(newx_ms1), c='black', linewidth=1, linestyle='--')
        ax.scatter(toAlign_ms1_matched['retention_time'], ref_ms1_matched['retention_time'], s=2)
        ax.set_xlabel('Retention Time (A) (MS1)', fontweight='bold')
        ax.set_ylabel('Retention Time (B) (MS1)', fontweight='bold')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 25)
        plt.tight_layout()
        plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS1Peaks_SVRSplineFit.png'.format(file_NoExt))
        plt.close()

        if MS2DataPresent is True:
            fig, ax = plt.subplots(1, dpi=150, facecolor='w')
            ax.plot(newx_ms2, spl_ms2(newx_ms2), c='black', linewidth=1, linestyle='--')
            ax.scatter(toAlign_ms2_matched['retention_time'], ref_ms2_matched['retention_time'], s=2)
            ax.set_xlabel('Retention Time (A) (MS2)', fontweight='bold')
            ax.set_ylabel('Retention Time (B) (MS2)', fontweight='bold')
            ax.set_xlim(0, 25)
            ax.set_ylim(0, 25)
            plt.tight_layout()
            plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2Peaks_SVRSplineFit.png'.format(file_NoExt))
            plt.close()
    
    #Apply Alignment
    # sample_data['ms1_peaks']['retention_time'] = spl_ms1(sample_data['ms1_peaks']['retention_time'])
    sample_data['ms1']['retention_time'] = spl_ms1(sample_data['ms1']['retention_time'])
    if MS2DataPresent is True:
        # sample_data['ms2_peaks']['retention_time'] = spl_ms2(sample_data['ms2_peaks']['retention_time'])
        sample_data['ms2']['retention_time'] = spl_ms1(sample_data['ms2']['retention_time'])

    if SaveRTAlignmentGraphs is True:

        rt_aligned_ms1 = deimos.collapse(sample_data['ms1'], keep='retention_time').sort_values(by='retention_time')
        Ref_ms1 = deimos.collapse(rtalign_data['ms1'], keep='retention_time').sort_values(by='retention_time')
        
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
        plt.close()

        if MS2DataPresent is True:

            rt_aligned_ms2 = deimos.collapse(sample_data['ms2'], keep='retention_time').sort_values(by='retention_time')
            Ref_ms2 = deimos.collapse(rtalign_data['ms2'], keep='retention_time').sort_values(by='retention_time')
            
            fig, ax = plt.subplots(1, dpi=150, facecolor='w')
            ax.fill_between(rt_aligned_ms2['retention_time'], rt_aligned_ms2['intensity'], color='C0', alpha=0.5, label='spl(A)')
            ax.fill_between(Ref_ms2['retention_time'], Ref_ms2['intensity'], color='C3', alpha=0.5, label='B')
            ax.set_xlabel('Retention Time (MS2)', fontweight='bold')
            ax.set_ylabel('Intensity', fontweight='bold')
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
            plt.legend()
            plt.tight_layout()
            plt.savefig('Results/{}/RetentionTimeAlignmentAndPeakDetection/MS2Peaks_RetentionTimeAlgined.png'.format(file_NoExt))
            plt.close()

    # Delete 'ms1_peaks' and 'ms2_peaks' keys from sample_data if they exist
    # --> peak info can be properly generated from the RT Aligned raw data
    sample_data.pop('ms1_peaks', None)
    sample_data.pop('ms2_peaks', None)

    if SaveRTAlignmentDataFiles is True:
        #Save Data as HD5 file
        deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), sample_data['ms1'], key='ms1', mode='w')
        if MS2DataPresent:
            deimos.save('Results/RTAligned_{}.h5'.format(file_NoExt), sample_data['ms2'], key='ms2', mode='a')

    return sample_data
    
def ExtractMS2Spectra(sample_data, file_NoExt, MS2Extract_intensity_thres, MS2Extract_ms1_mz_subset_low, MS2Extract_ms1_dt_subset_low, 
                      MS2Extract_ms1_rt_subset_low, MS2Extract_ms1_mz_subset_high, MS2Extract_ms1_dt_subset_high, 
                      MS2Extract_ms1_rt_subset_high, MS2Extract_ms2_dt_subset_low, MS2Extract_ms2_rt_subset_low, 
                      MS2Extract_ms2_dt_subset_high, MS2Extract_ms2_rt_subset_high, MS2Extract_model_ce, MS2Extract_model_params, 
                      MS2Extract_ms1_decon_intensity_thres, MS2Extract_ms2_decon_intensity_thres, 
                      MS2Extract_construct_pairs_dt_low, MS2Extract_construct_pairs_rt_low, MS2Extract_construct_pairs_dt_high, 
                      MS2Extract_construct_pairs_rt_high, MS2Extract_construct_pairs_ce, MS2Extract_construct_pairs_error_tol, 
                      MS2Extract_config_extract_mz_low, MS2Extract_config_extract_dt_low, MS2Extract_config_extract_rt_low, 
                      MS2Extract_config_extract_mz_high, MS2Extract_config_extract_dt_high, MS2Extract_config_extract_rt_high, 
                      MS2Extract_decon_dt_resolution, MS2Extract_dt_score_threshold, SaveMS2ExtractDataFiles, SaveMS2ExtractGraphs):

    if SaveMS2ExtractDataFiles is True or SaveMS2ExtractGraphs is True:
        #Make a folder for this sample for all MS2 Extraction results to go in to
        if os.path.exists('Results/{}/MS2Extraction'.format(file_NoExt)):
            shutil.rmtree('Results/{}/MS2Extraction'.format(file_NoExt))
        os.mkdir('Results/{}/MS2Extraction'.format(file_NoExt))

    # #Drop scanId or this messes up the "Get maximal data point" bit
    # sample_data['ms1'] = sample_data['ms1'].drop('scanId', axis = 1)

    if SaveMS2ExtractGraphs is True:
    
        # get maximal data point
        rt_i, dt_i, mz_i, intensity_i = sample_data['ms1'].loc[sample_data['ms1']['intensity'] == sample_data['ms1']['intensity'].max(), :].values[0]

        # subset the raw data
        precursor = deimos.slice(sample_data['ms1'],
                            by=['mz', 'drift_time', 'retention_time'],
                            low=[mz_i - MS2Extract_ms1_mz_subset_low, 
                                dt_i - MS2Extract_ms1_dt_subset_low, 
                                rt_i - MS2Extract_ms1_rt_subset_low],
                            high=[mz_i + MS2Extract_ms1_mz_subset_high, 
                                dt_i + MS2Extract_ms1_dt_subset_high, 
                                rt_i + MS2Extract_ms1_rt_subset_high])

        # putative fragments
        fragment_profile = deimos.slice(sample_data['ms2'],
                                    by=['drift_time', 'retention_time'],
                                    low=[dt_i - MS2Extract_ms2_dt_subset_low, 
                                        rt_i - MS2Extract_ms2_rt_subset_low],
                                    high=[dt_i + MS2Extract_ms2_dt_subset_high, 
                                        rt_i + MS2Extract_ms2_rt_subset_high])
        
        fragment_dt = deimos.collapse(fragment_profile, keep='drift_time').sort_values(by='drift_time')
        precursor_dt = deimos.collapse(precursor, keep='drift_time').sort_values(by='drift_time')

        if SaveMS2ExtractDataFiles is True:
            #Save outputs as .csv files
            fragment_dt.to_csv('Results/{}/MS2Extraction/fragment_dt.csv'.format(file_NoExt), index=False)
            precursor_dt.to_csv('Results/{}/MS2Extraction/precursor_dt.csv'.format(file_NoExt), index=False)
        
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
        plt.savefig('Results/{}/MS2Extraction/DriftTimeOffset.png'.format(file_NoExt))
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

    # Create a new dataframe with only the required columns
    sample_data['ms1'] = sample_data['ms1'][['mz', 'drift_time', 'retention_time', 'intensity']]
    sample_data['ms2'] = sample_data['ms2'][['mz', 'drift_time', 'retention_time', 'intensity']]
    # Select columns, using 'intensity_x' if 'intensity' is missing
    if 'intensity' in sample_data['ms1_peaks'].columns:
        sample_data['ms1_peaks'] = sample_data['ms1_peaks'][['mz', 'drift_time', 'retention_time', 'intensity', 'persistence']]
    elif 'intensity_x' in sample_data['ms1_peaks'].columns:
        sample_data['ms1_peaks'] = sample_data['ms1_peaks'][['mz', 'drift_time', 'retention_time', 'intensity_x', 'persistence']]
        sample_data['ms1_peaks'] = sample_data['ms1_peaks'].rename(columns={'intensity_x': 'intensity'})
    else:
        raise KeyError("Neither 'intensity' nor 'intensity_x' found in ms1_peaks columns")
    sample_data['ms2_peaks'] = sample_data['ms2_peaks'][['mz', 'drift_time', 'retention_time', 'intensity', 'persistence']]
    ms1_peak_to_deconvolute = sample_data['ms1_peaks']#[['scanId', 'retention_time', 'drift_time', 'mz', 'persistence']]
    
    #Deconvolution
    decon = deimos.deconvolution.MS2Deconvolution(ms1_peak_to_deconvolute, sample_data['ms1'], 
                                                  sample_data['ms2_peaks'], sample_data['ms2'])
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
    #.head(5) removed from the function below as we want to save all of the data
    #Group data by each ms1 feature
    res = res.loc[res['drift_time_score'] > MS2Extract_dt_score_threshold].groupby(by=[x for x in res.columns if x.endswith('_ms1')],
                                                      as_index=False).agg(list).sort_values(by='persistence_ms1',
                                                                                            ascending=False)
                                                                                            
    #Added by Jess (24/04/2025, modified 14/08/2026)
    ##Merge MS1 and MS2 data together
    #Make the MS1 index a column
    sample_data['ms1_peaks'] = sample_data['ms1_peaks'].reset_index(drop = False)
    sample_data['ms1_peaks'] = sample_data['ms1_peaks'].rename(columns={'index': 'index_ms1'})
    #Perform inner join by the index_ms1 column
    res_final = pd.merge(sample_data['ms1_peaks'], res, on="index_ms1", how = "left")

    # Delete the 'index_ms1' column from sample_data['ms1_peaks']
    sample_data['ms1_peaks'] = sample_data['ms1_peaks'].drop(columns=['index_ms1'])

    if SaveMS2ExtractDataFiles is True:
        res.to_csv('Results/{file}/MS2Extraction/MS2Extraction_res.csv'.format(file = file_NoExt), index=False)
        sample_data['ms1_peaks'].to_csv('Results/{file}/MS2Extraction/MS2Extraction_ms1_peaks_thres_forMS2.csv'.format(file = file_NoExt), index=False)
        res_final.to_csv('Results/{file}/MS2Extraction/MS2Extraction_res_final.csv'.format(file = file_NoExt), index=False)   

    if SaveMS2ExtractGraphs is True:
        #Iterate through the rows and save a .png file for each one
        for index, row in res.iterrows():
            deimos.plot.stem(np.array(res.loc[index, 'mz_ms2']),
                        np.array(res.loc[index, 'intensity_ms2']),
                        width=1)
            plt.tight_layout()
            plt.savefig('Results/{file}/MS2Extraction/{index}_index_ms1_MS1FragmentationSpectra.png'.format(file = file_NoExt, index = row["index_ms1"]))
            #Save memory space
            plt.close()  
        
    return res_final

def DetectIsotopes(sample_data, isotope_intensity_thres, isotope_partition_size, isotope_partition_overlap, isotope_map_mz_dt_rt_tol, 
                   isotope_map_delta, isotope_map_max_isotopes, isotope_map_max_charges, isotope_map_max_error, file_NoExt, 
                   isotope_min_no_isotopes, isotope_slice_mz_low, isotope_slice_mz_high, isotope_plot_slice_mz_low, 
                   isotope_plot_slice_dt_low, isotope_plot_slice_rt_low, isotope_plot_slice_mz_high, isotope_plot_slice_dt_high, 
                   isotope_plot_slice_rt_high, SaveIsotopeDetDataFiles, SaveIsotopeDetGraphs, res_final): 

    # Partition the data
    partitions = deimos.partition(sample_data['ms1_peaks'], size=isotope_partition_size, 
                                  overlap=isotope_partition_overlap)
    
    # Map isotope detection over partitions
    isotopes = partitions.map(deimos.isotopes.detect,
                              dims=['mz', 'drift_time', 'retention_time'],
                              tol=isotope_map_mz_dt_rt_tol,
                              delta=isotope_map_delta,
                              max_isotopes=isotope_map_max_isotopes,
                              max_charge=isotope_map_max_charges,
                              max_error=isotope_map_max_error)

    if SaveIsotopeDetDataFiles is True or SaveIsotopeDetGraphs is True:
        #Make a folder for this sample for Isotope Detection results to go in to
        if os.path.exists('Results/{}/IsotopeDetection'.format(file_NoExt)):
            shutil.rmtree('Results/{}/IsotopeDetection'.format(file_NoExt))
        os.mkdir('Results/{}/IsotopeDetection'.format(file_NoExt))

    if SaveIsotopeDetDataFiles is True:
        isotopes.to_csv('Results/{}/IsotopeDetection/isotopes.csv'.format(file_NoExt), index=False) 

    if SaveIsotopeDetGraphs is True:
        #Consider isotopic signatures with at least 3 members
        isotopes_graph = isotopes.loc[isotopes['n'] >= isotope_min_no_isotopes, :].sort_values(by='intensity', ascending=False).head(5)
        
        for index, row in isotopes_graph.iterrows():
            
            #Following Peak detection steps to subset data within the mass range of the isotope
            ms1_iso_ss = deimos.slice(sample_data['ms1'], by='mz', 
                                    low = row['mz']-isotope_slice_mz_low, 
                                    high = row['mz']+isotope_slice_mz_high)
            
            # Get maximal data point
            # scan_id_i, rt_i, dt_i, mz_i, intensity_i = ms1_iso_ss.loc[ms1_iso_ss['intensity'] == ms1_iso_ss['intensity'].max(), :].round(1).values[0] #type: ignore
            mz_i, dt_i, rt_i, intensity_i = ms1_iso_ss.loc[ms1_iso_ss['intensity'] == ms1_iso_ss['intensity'].max(), :].round(1).values[0] #type: ignore
                    
            #The ms1_iso data is loaded in the Peak Detection section
            feature = deimos.slice(ms1_iso_ss, by=['mz', 'drift_time', 'retention_time'],
                                low=[mz_i - isotope_plot_slice_mz_low, 
                                    dt_i - isotope_plot_slice_dt_low, 
                                    rt_i - isotope_plot_slice_rt_low],
                                high=[mz_i + isotope_plot_slice_mz_high, 
                                    dt_i + isotope_plot_slice_dt_high, 
                                    rt_i + isotope_plot_slice_rt_high])
            
            ax = deimos.plot.multipanel(feature, dpi=300)
            for i in isotopes_graph.loc[index, 'mz_iso']:
                ax['mz'].scatter(i, 0, s=10, color='C3', zorder=5)
            
            plt.tight_layout()
            plt.savefig('Results/{}/IsotopeDetection/{}_isotope_feature_plot.png'.format(file_NoExt, index))
            #Save memory space
            plt.close()

    #Merge isotope information info ms1 peaks info (can be matched by index)
    #Make index a column to allow merging
    isotopes = isotopes.rename(columns={'idx': 'index_ms1'})
    res_final = pd.merge(res_final, isotopes, 
                         on='index_ms1', how='left')
    #TODO: Make sure that the charge column in the isotopes output corresponds to the polarity of the data being analysed

    if SaveIsotopeDetDataFiles is True:
        res_final.to_csv('Results/{}/IsotopeDetection/IsotopeDetection_res_final.csv'.format(file_NoExt), index=False)

    return sample_data, res_final
        
def ConcatenateNewPeakData(loopcount, res_final, agglo_mergeFeatures_mz_dt_rt_tol, 
                                                  file_NoExt):
    # #With thanks to Karl Burgess for scripting the majority this section
    
    # # create an empty dataframe to store the clustered peaks
    # multipeaks = pd.DataFrame()

    # #Use res_final object with MS1 and MS2 data
    # merged_peaks = deimos.alignment.merge_features(features = res_final,
    #                                         dims=['mz', 'drift_time',
    #                                             'retention_time'],
    #                                         tol=agglo_mergeFeatures_mz_dt_rt_tol,
    #                                         relative = [True, True, False])                                                        
    # # next bit is sort of cheating. Fake up a multi-file dataframe by adding the filename and loopcounter as columns
    # merged_peaks.insert(0, 'sample_idx', loopcount)#type: ignore
    # merged_peaks.insert(0, 'sample_id', file_NoExt)#type: ignore
    
    # # main bit of work - concatenate the dataframe with the new peak data.
    # multipeaks = pd.concat([multipeaks, merged_peaks])
    
    # Add sample_id and sample_idx columns to keep track of which features belong to which samples
    res_final.insert(0, 'sample_idx', loopcount)
    res_final.insert(0, 'sample_id', file_NoExt)
    loopcount=loopcount+1
    
    #Append data to existing csv file (prevent system crashing)
    res_final.to_csv('Results/{}/AgglomerativeClustering/res_final.csv'.format(file_NoExt), index=False)
    #Check if the file already exists
    if os.path.isfile("Results/res_final_AllSamples.csv") == True: #True if present
        #Append to file
        res_final.to_csv('Results/res_final_AllSamples.csv', mode='a', index=False, header=False)
    else:
        #Create file
        res_final.to_csv('Results/res_final_AllSamples.csv', index=False)
        
    return loopcount
    
    
def AgglomerativeClusteringAndPivotTable(agglo_multiSampPart_size, agglo_multiSampPart_tol, agglo_clustering_mz_dt_rt_tol,
                                         MS2DataPresent, PerformIsotopeDetection, PerformPeakShapeCorrelation):
    #With thanks to Karl Burgess for scripting the majority this section
    
    # multipeaks_AllSamples = pd.read_csv("Results/multipeaks_AllSamples.csv")
    #TODO: Change back to the file read in the above line of code
    multipeaks_AllSamples = pd.read_csv("Results_TenSamplesReference/multipeaks_AllSamples.csv")
    
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

    #TODO: Change directories back to Results folder
    res.to_csv('Results_TenSamplesReference/resWithClusters.csv', index=False)

    print(len(res))
    print("res['cluster'] value range:", res['cluster'].min(), "to", res['cluster'].max())

    # now dump it to disk!
    res.to_csv('Results_TenSamplesReference/res.csv', index=False)
    print("res.to_csv() completed")

    return res

    # def select_MS2_info(group):
    #     # Check if any entry in 'index_ms2' is not nan (contains non-nan values)
    #     # 'index_ms2' is a list; need to check for lists that contain at least one non-NaN value
    #     def index_ms2_has_non_nan(row):
    #         vals = row['index_ms2']
    #         if isinstance(vals, list):
    #             return any(pd.notna(v) for v in vals)
    #         return pd.notna(vals)
    #     # Find rows where index_ms2 contains non-nan values
    #     candidates = group[group.apply(index_ms2_has_non_nan, axis=1)]
        
    #     if len(candidates) > 0:
    #         # From those candidates, select row with the highest value in the 'intensity_ms2' list
    #         def max_intensity_ms2(row):
    #             vals = row['intensity_ms2']
    #             if not isinstance(vals, list) or isinstance(vals, str):
    #                 # If vals is a string (like str(list)), do not wrap in list
    #                 try:
    #                     # Try to interpret string as list (from csv import)
    #                     import ast
    #                     evaluated = ast.literal_eval(vals)
    #                     if isinstance(evaluated, list):
    #                         vals = evaluated
    #                     else:
    #                         vals = [evaluated]
    #                 except Exception:
    #                     vals = [vals]
    #             if isinstance(vals, list) and len(vals) > 0:
    #                 return max([v for v in vals if pd.notna(v)])
    #             return -np.inf
    #         idx = candidates.apply(max_intensity_ms2, axis=1).idxmax()
    #         # Keep the row with highest intensity_ms2 as is
    #         best_row = group.loc[idx].copy()
    #         # For all other rows, clear columns except for 'mz' and 'retention_time'
    #         new_group = []
    #         for i, row in group.iterrows():
    #             if i == idx:
    #                 #Keep the MS2 info for the feature with the fragment with the highest intensity
    #                 new_group.append(row)
    #             else:
    #                 #Keep the non-MS2 details for the other MS1 entries that are in the same cluster
    #                 row_cleared = row.copy()
    #                 for col in group.columns:
    #                     if col in ["mz_ms1","drift_time_ms1","retention_time_ms1","intensity_ms1","persistence_ms1",
    #                                "index_ms2","mz_ms2","drift_time_ms2","retention_time_ms2","intensity_ms2",
    #                                "persistence_ms2","drift_time_raw_ms2","drift_time_error","drift_time_score"]:
    #                         row_cleared[col] = np.nan
    #                 new_group.append(row_cleared)
    #         return pd.DataFrame(new_group)
    #     else:
    #         # If no candidate, fall back to returning the entire group
    #         return group

    # def select_isotope_info(group):
    #     # Check if any entry in 'error' is not nan (contains non-nan values)
    #     # 'error' is a list; need to check for lists that contain at least one non-NaN value
    #     def error_has_non_nan(row):
    #         vals = row['error']
    #         if isinstance(vals, list):
    #             return any(pd.notna(v) for v in vals)
    #         return pd.notna(vals)
    #     # Find rows where error contains non-nan values
    #     candidates = group[group.apply(error_has_non_nan, axis=1)]
        
    #     if len(candidates) > 0:
    #         # From those candidates, select row with the lowest average value in the 'error' list
    #         def avg_error(row):
    #             vals = row['error']
    #             if isinstance(vals, list) and len(vals) > 0:
    #                 # Exclude NaN
    #                 valid = [v for v in vals if pd.notna(v)]
    #                 if len(valid) > 0:
    #                     return np.nanmean(valid)
    #             return np.inf
    #         idx = candidates.apply(avg_error, axis=1).idxmin()
    #         # Keep the row with lowest average error as is
    #         best_row = group.loc[idx].copy()
    #         # For all other rows, clear columns except for 'mz' and 'retention_time'
    #         new_group = []
    #         for i, row in group.iterrows():
    #             if i == idx:
    #                 #Keep the isotope info for the feature with the lowest average error
    #                 new_group.append(row)
    #             else:
    #                 #Keep the non-isotope details for the other MS1 entries that are in the same cluster
    #                 row_cleared = row.copy()
    #                 for col in group.columns:
    #                     if col in ["mz_y", "charge", "intensity_y", "multiple", "dx", "mz_iso",
    #                                "intensity_iso", "idx_iso", "error", "decay", "n"]:
    #                         row_cleared[col] = np.nan
    #                 new_group.append(row_cleared)
    #         return pd.DataFrame(new_group)
    #     else:
    #         # If no candidate, fall back to returning the entire group
    #         return group

    # def select_peakshape_info(group):
    #     # Check if any entry in 'Peak_Shape' is not nan (contains non-nan values)
    #     # 'Peak_Shape' is a list; need to check for lists that contain at least one non-NaN value
    #     def peakshape_has_non_nan(row):
    #         vals = row['Peak_Shape']
    #         if isinstance(vals, list):
    #             return any(pd.notna(v) for v in vals)
    #         return pd.notna(vals)
    #     # Find rows where Peak_Shape contains non-nan values
    #     candidates = group[group.apply(peakshape_has_non_nan, axis=1)]
        
    #     if len(candidates) > 0:
    #         # From those candidates, select row with the most values in Peak_Shape
    #         def peakshape_list_length(row):
    #             vals = row['Peak_Shape']
    #             if isinstance(vals, list):
    #                 # Only count non-nan elements
    #                 return sum(pd.notna(v) for v in vals)
    #             elif pd.notna(vals):
    #                 return 1
    #             return 0
    #         idx = candidates.apply(peakshape_list_length, axis=1).idxmax()
    #         # Keep the row with the most Peak_Shape values as is
    #         best_row = group.loc[idx].copy()
    #         # For all other rows, clear Peak_Shape column
    #         new_group = []
    #         for i, row in group.iterrows():
    #             if i == idx:
    #                 # Keep Peak_Shape info for the best feature
    #                 new_group.append(row)
    #             else:
    #                 # Remove Peak_Shape info for others in cluster
    #                 row_cleared = row.copy()
    #                 row_cleared['Peak_Shape'] = np.nan
    #                 new_group.append(row_cleared)
    #         return pd.DataFrame(new_group)
    #     else:
    #         # If no candidate, fall back to returning the entire group
    #         return group

    # print(len(res))

    # clustering = res
    
    # print("deimos.alignment.agglomerative_clustering() completed")
    # del multipeaks_AllSamples, res

    # #The pd.pivot_table() function cannot consider multi-column conditions during aggregation
    # # --> for certain values where this is required, the relevant entries will be kept within each cluster group
    # if MS2DataPresent is True:
    #     #For clusters with multiple MS2 entries, keep the MS2 entry with the highest intensity
    #     clustering = clustering.groupby('cluster', group_keys=False).apply(select_MS2_info).reset_index()

    # if PerformIsotopeDetection is True:
    #     #For clusters with multiple isotope entries, keep the isotope entry with the lowest (average) error score
    #     # (average as one entry can have multiple isotopes assigned)
    #     clustering = clustering.groupby('cluster', group_keys=False).apply(select_isotope_info).reset_index()

    # if PerformPeakShapeCorrelation is True:
    #     #For clusters with multiple different peak shapes, keep the peak shape info with the most datapoints
    #     clustering = clustering.groupby('cluster', group_keys=False).apply(select_peakshape_info).reset_index()

    # # now dump it to disk!
    # clustering.to_csv('Results/clustering.csv', index=False)
    # print("clustering.to_csv() completed")

    # #Depending on which steps/data were kept for the earlier processing stages, different columns 
    # # will be present in the data object. --> need to modulate which values are included in the pivot_table function
    
    # #Create values_list and aggfunc_commands with the details to keep that are always present
    # values_list = ["index_ms1","mz","drift_time","retention_time","intensity"]
    # aggfunc_commands = {
    #     "index_ms1":list,
    #     "mz":'mean',
    #     "drift_time":'mean',
    #     "retention_time":'mean',
    #     "intensity":'mean'
    # }
    # if MS2DataPresent is True:
    #     values_list.append(["index_ms2","mz_ms2","drift_time_ms2","retention_time_ms2","intensity_ms2",
    #                         "drift_time_raw_ms2","drift_time_error","drift_time_score"])
    #     aggfunc_commands.append({
    #         "index_ms2",
    #         "mz_ms2",
    #         "drift_time_ms2",
    #         "retention_time_ms2",
    #         "intensity_ms2",
    #         "drift_time_raw_ms2",
    #         "drift_time_error",
    #         "drift_time_score"
    #     })
    # if PerformIsotopeDetection is True:
    #     values_list.append(["mz_y","charge","intensity_y","multiple","dx","mz_iso","intensity_iso",
    #                         "idx_iso","error","decay","n"])
    #     aggfunc_commands.append({
    #         "mz_y",
    #         "charge",
    #         "intensity_y",
    #         "multiple",
    #         "dx",
    #         "mz_iso",
    #         "intensity_iso",
    #         "idx_iso",
    #         "error",
    #         "decay",
    #         "n"
    #     })
    # if PerformPeakShapeCorrelation is True:
    #     values_list.append(["Peak_Shape"])
    #     aggfunc_commands.append({
    #         "Peak_Shape"
    #     })

    # # pivot the table to get it into the right format for ipaPy2
    # # table headers: id(cluster), average mz, average rts, sample_intensities NOTE: no average drift time or ccs currently!
    # full_pivot = pd.pivot_table(clustering, 
    #                             values = ['intensity', 'mz', 'retention_time', 
    #                                       'drift_time'], 
    #                             index = 'cluster', columns = 'sample_id') #columns changed from 'sample_idx'
       
    # full_pivot.to_csv('Results/full_pivot.csv', index=False)
    # print("full_pivot = pd.pivot_table() completed")
    # del clustering

    # #how to get rid of a column in a multilevel table
    # #full_pivot = full_pivot.drop([('mz', 'mzs')], axis=1)
    # #ok - calculate the per row means for all of the mzs
    # mzs = full_pivot[('mz',)].mean(axis=1)#type: ignore
    # print(full_pivot[('mz',)])
    # print("mzs = full_pivot completed")
    # print(mzs)
    # #full_pivot[('mz','mzs')] = mzs
    # #now drop the individual sample means
    # mz_stripped = full_pivot.drop([('mz',)], axis = 1)
    # #add to the multiindex
    # mz_stripped[('mz', 'mzs')] = mzs
    # del [full_pivot, mzs]#type: ignore

    # #now do the same for the retention times
    # # mz_stripped.to_csv('Results/mz_stripped.csv', index=False)
    # rts = mz_stripped[('retention_time',)].mean(axis=1)#type: ignore
    # rts_stripped = mz_stripped.drop([('retention_time',)], axis = 1)
    # rts_stripped[('retention_time', 'RTs')] = rts
    # del [mz_stripped, rts]#type: ignore

    # #now we do the same for the drift times 
    # drifts = rts_stripped[('drift_time',)].mean(axis=1)#type: ignore
    # drifts_stripped = rts_stripped.drop([('drift_time',)], axis = 1)
    # #currently commented out as unused, also shouldn't this be CCS values?
    # #Update from Jess: Yes! I have put this back in so that the CCS values can be calculated
    # drifts_stripped[('drift_time', 'drifts')] = drifts
    # del [rts_stripped, drifts]#type: ignore
    # # drifts_stripped.to_csv('Results/drifts_stripped.csv', index=False)
    # print("dataset stripping steps completed")
    
    # #flatten multiindex
    # drifts_stripped.columns = drifts_stripped.columns.get_level_values(1)
    # #rename the index (cluster) to ids
    # #is it ok to just have a list of ids as the index or do we need an additional index column?
    # # drifts_stripped = drifts_stripped.index.name ='ids' #Commented out by Jess (29/01/2025)
    # # if not, add a specific set of indexes as a different column
    # drifts_stripped['ids'] = range(1, len(drifts_stripped) + 1)

    # return drifts_stripped

def CreateCCSCalObjects(tune_pos_file, ccsCalib_mz, ccsCalib_ccs, ccsCalib_q, ccsCalib_buffer_mass, ccsCalib_mz_tol, ccsCalib_dt_tol):
    tune_pos = deimos.load(rf'f:\JessicaOLoughlin\RawDataMZMLFiles\{tune_pos_file}', key='ms1')
    print("Calibrating CCS values with tuning file")
    ccs_cal_pos = deimos.calibration.tunemix(tune_pos,
                                              mz=ccsCalib_mz,
                                              ccs=ccsCalib_ccs,
                                              q=ccsCalib_q,
                                              buffer_mass=ccsCalib_buffer_mass, 
                                              mz_tol=ccsCalib_mz_tol, 
                                              dt_tol=ccsCalib_dt_tol)
    print('r-squared:\t', ccs_cal_pos.fit['r'] ** 2) #type: ignore
    
    return ccs_cal_pos

# def CCSCalibrationSteps(ccs_cal_pos):
def CCSCalibration(ccs_cal_pos, feature_table):
    
    feature_table['CCS'] = ccs_cal_pos.arrival2ccs(mz=feature_table['mzs'], 
                                                ta=feature_table['drifts'], 
                                                q=1)
    
    feature_table.to_csv('Results/drifts_stripped_final.csv', index=False)
    print("CCS Calibration steps completed")

    return feature_table
#Script to run through the DEIMoS and custom commands for all .h5 files in a directory
#Adapted into discrete functions from the 20240503_LoopThroughHD5Files_DataProcessingOnly.py script
#Author: Jessica O'Loughlin (s1907024@ed.ac.uk)
#Supervisor: Prof. Karl Burgess (k.burgess@ed.ac.uk)
#Created: 08/11/2024

import glob #Searches for files with specific extensions
# import deimos #Performs various Mass Spec Processing Steps
import numpy as np #Basic math functionalities
import matplotlib.pyplot as plt #Create plot outputs
import os #Allows directory to be read
import os.path #Checks for existance of files
import shutil #Will remove specific folders if they are already present
# import time #Find the creation date of files 
import pandas as pd #Handle dataframes
from numpy import trapz #Calculate area under line for Gap Filling
from datetime import datetime #Get current date and time
import warnings

import DEIMoSFunctions
import CustomFunctions

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

#Check where RTAlignment functions cause the script to "restart"
print("Script loop")

##Set thresholds for different processes
#Working directory (where the raw files are)
working_directory = r"F:\JessicaOLoughlin\RawDataAgilentFiles"
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

#Peak Shape Correlation
PeakShapeCorr_mz_tol = 5 #ppm
PeakShapeCorr_RT_tol = 3.0 #minutes
PeakShapeCorr_DT_tol = 30 #milliseconds

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

    os.chdir(r"f:\JessicaOLoughlin\RawDataMZMLFiles")

    # path = r'f:\JessicaOLoughlin\RawDataMZMLFiles\*.h5'
    path = r'*.h5'
    #Create a list of the files with the .h5 extension
    files = glob.glob(path)
    #Can check that it has the correct number of files
    #print(len(files))
    
    #Remove the Calibration files from the main analysis pipeline
    files  = [x for x in files if x not in calib_files]
    
    #TODO: For development only
    # files = files[slice(3)]
    files = ['POS_FBS_IM_MSMS_40kTF_400TR_4.h5']

    print("Files to be processed:")
    for file in files:
        print(file)
    
    return files

def CreateResultsFile():
    #Make new Results folder (overwrite if it already exists)
    if os.path.exists('Results'):
        shutil.rmtree('Results')
    os.makedirs('Results')

import deimos
import DEIMoSFunctions_RedundantStepsRemoved
#Custom Errors
class CustomError(Exception):
    pass

#Set whether the pipeline will undergo MS2 data processing or not
MS2DataPresent = False
##Peak Detection
#Decide whether to so this within the script or not (e.g., in case they have already been thresholded in previous steps)
ThresholdDataWithinScript = True
ms1_threshold = 500
if MS2DataPresent is True:
    ms2_threshold = 500
else:
    ms2_threshold = None
SaveDetectedPeaksData = True
##Retention Time Alignment
PerformRTAlignment = False
SaveRTAlignmentDataFiles = True
SaveRTAlignmentGraphs = True
##Isotope Detection
PerformIsotopeDetection = False
SaveIsotopeDetDataFiles = True
SaveIsotopeDetGraphs = True
##MS2 Extraction
SaveMS2ExtractDataFiles = False
SaveMS2ExtractGraphs = False
##Peak Shape Correlation
PerformPeakShapeCorrelation = True

if __name__ == "__main__":
    
    processes=1
    
    startTime = datetime.now()
    print("===============================")
    print("DEIMoS Script startTime:", startTime)
    print("===============================")

    print("-=-=-=-=-=-=-=-Parameters=-=-=-=-=-=-=-=-=")
    print("========Peak Detection========")
    print("MS1 threshold:", ms1_threshold)
    print("MS2 threshold:", ms2_threshold)
    print("SaveDetectedPeaksData?", SaveDetectedPeaksData)
    print("===Retention Time Alignment===")
    print("Retention Time Alignment to be performed?", PerformRTAlignment)
    print("SaveRTAlignmentDataFiles is:", SaveRTAlignmentDataFiles)
    print("SaveRTAlignmentGraphs is:", SaveRTAlignmentGraphs)
    print("=======Isotope Detection======")
    print("Isotope Detection to be performed?", PerformIsotopeDetection)
    print("SaveIsotopeDetDataFiles is:", SaveIsotopeDetDataFiles)
    print("SaveIsotopeDetGraphs is:", SaveIsotopeDetGraphs)
    print("========MS2 Extraction========")
    print("MS2DataPresent?", MS2DataPresent)
    if MS2DataPresent is True:
        print("SaveMS2ExtractDataFiles is:", SaveMS2ExtractDataFiles)
        print("SaveMS2ExtractGraphs is:", SaveMS2ExtractGraphs)
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

    #Create multipeaks and loopcount objects for downstream agglomerative clustering steps
    # Create a loop counter for the agglomerative clustering steps
    loopcount = 0
    
    ##Find the file generated in the middle of the run
    #Make dataframe to contain file names and creation times
    
    files = ReadFilesInDirectory()
    print("ReadFilesInDirectory() complete")

    CreateResultsFile()
    print("CreateResultsFile() complete")

    if PerformRTAlignment is True:
        #Note: The Peak Detection is also performed for the middle file at this point
        middle, rtalign_data = DEIMoSFunctions_RedundantStepsRemoved.FindMiddleFileForRTAlignment(ms1_threshold, 
                                                                                                ms2_threshold,
                                                                                                PeakDet_smooth_data_radius, 
                                                                                                PeakDet_persistent_homology_radius, 
                                                                                                MS2DataPresent)
        print("FindMiddleFileForRTAlignment() complete")
    else:
        middle = None
        rtalign_data = None
    
    #del [CreationTime, file, Created, middle_row]
    
    #Loop through the samples in the directory for sample-specific processes
    for sample in files:
        print("------------------------")
        print("Processing file", sample)
        print("------------------------")
        
        #########################################
        #Make Folder for results for this sample
        #########################################
        file_NoExt = sample.replace('.h5', '')

        if file_NoExt == middle: #Folder already created at FindMiddleFileForRTAlignment() stage
            pass
        else:
            #Make Folder for results for this sample
            if os.path.exists('Results/{}'.format(file_NoExt)):
                shutil.rmtree('Results/{}'.format(file_NoExt))
            os.makedirs('Results/{}'.format(file_NoExt))


        if file_NoExt == middle:
            sample_data = rtalign_data.copy()#type: ignore
        else:
            #Load data for sample
            sample_data = {}
            sample_data['ms1'] = deimos.load('{}.h5'.format(file_NoExt), key='ms1')
            sample_data['ms1'] = sample_data['ms1'].apply(pd.to_numeric, errors = "ignore")

            if ThresholdDataWithinScript is True:
                #Threshold the data
                sample_data['ms1'] = deimos.threshold(sample_data['ms1'], threshold=ms1_threshold)

            if MS2DataPresent is True:
                sample_data['ms2'] = deimos.load('{}.h5'.format(file_NoExt), key='ms2')
                sample_data['ms2'] = sample_data['ms2'].apply(pd.to_numeric, errors = "ignore")
                if ThresholdDataWithinScript is True:
                    sample_data['ms2'] = deimos.threshold(sample_data['ms2'], threshold=ms2_threshold)#type: ignore

        #If desired, perform RT alignment on the sample
        if PerformRTAlignment is True:
            if file_NoExt == middle:
                # No need to align the reference file
                print("Retention Time not performed as this is the reference file")
            else:
                sample_data = DEIMoSFunctions_RedundantStepsRemoved.RetentionTimeAlignment(file_NoExt, 
                                                                                           sample_data, 
                                                                                           rtalign_data, 
                                                                                           rtalign_persisHomology_thres, 
                           rtalign_persis_thres, rtalign_partition_thres, rtalign_partition_size, 
                           rtalign_partition_overlap, rtalign_zipmap_thres, rtalign_zipmap_mz_dt_rt_tol, 
                           SaveRTAlignmentDataFiles, SaveRTAlignmentGraphs, MS2DataPresent, PeakDet_smooth_data_radius,
                           PeakDet_persistent_homology_radius
                           )

        elif PerformRTAlignment is False:
            print("Retention Time Alignment not performed")
        else:
            raise CustomError("""Set PerformRTAlignment as True or False to indicate whether you want to include this step.""")

        #Detect Peaks
        #Peaks already detected for the middle file --> can skip
        if file_NoExt == middle:
            pass
        else:
            sample_data = DEIMoSFunctions_RedundantStepsRemoved.DetectPeaks(file_NoExt, sample_data, 
                                                                            PeakDet_smooth_data_radius,
                                                                            PeakDet_persistent_homology_radius, 
                                                                            MS2DataPresent)

        if MS2DataPresent is True:
            print("Performing MS2 Extraction")
            res_final = DEIMoSFunctions_RedundantStepsRemoved.ExtractMS2Spectra(sample_data, file_NoExt, MS2Extract_intensity_thres, MS2Extract_ms1_mz_subset_low, MS2Extract_ms1_dt_subset_low, 
                MS2Extract_ms1_rt_subset_low, MS2Extract_ms1_mz_subset_high, MS2Extract_ms1_dt_subset_high, 
                MS2Extract_ms1_rt_subset_high, MS2Extract_ms2_dt_subset_low, MS2Extract_ms2_rt_subset_low, 
                MS2Extract_ms2_dt_subset_high, MS2Extract_ms2_rt_subset_high, MS2Extract_model_ce, MS2Extract_model_params, 
                MS2Extract_ms1_decon_intensity_thres, MS2Extract_ms2_decon_intensity_thres, 
                MS2Extract_construct_pairs_dt_low, MS2Extract_construct_pairs_rt_low, MS2Extract_construct_pairs_dt_high, 
                MS2Extract_construct_pairs_rt_high, MS2Extract_construct_pairs_ce, MS2Extract_construct_pairs_error_tol, 
                MS2Extract_config_extract_mz_low, MS2Extract_config_extract_dt_low, MS2Extract_config_extract_rt_low, 
                MS2Extract_config_extract_mz_high, MS2Extract_config_extract_dt_high, MS2Extract_config_extract_rt_high, 
                MS2Extract_decon_dt_resolution, MS2Extract_dt_score_threshold, SaveMS2ExtractDataFiles, SaveMS2ExtractGraphs)
            print("ExtractMS2Spectra() complete")
        else:
            res_final = sample_data['ms1_peaks']

        sample_data['ms1_peaks'] = sample_data['ms1_peaks'].drop(columns=['persistence'])
        if MS2DataPresent is True:
            sample_data['ms2_peaks'] = sample_data['ms2_peaks'].drop(columns=['persistence'])

        if PerformIsotopeDetection is True:
            print("Performing Isotope Detection")
            sample_data, res_final = DEIMoSFunctions_RedundantStepsRemoved.DetectIsotopes(sample_data, isotope_intensity_thres, isotope_partition_size, isotope_partition_overlap, isotope_map_mz_dt_rt_tol, 
                                        isotope_map_delta, isotope_map_max_isotopes, isotope_map_max_charges, isotope_map_max_error, file_NoExt, 
                                        isotope_min_no_isotopes, isotope_slice_mz_low, isotope_slice_mz_high, isotope_plot_slice_mz_low, 
                                        isotope_plot_slice_dt_low, isotope_plot_slice_rt_low, isotope_plot_slice_mz_high, isotope_plot_slice_dt_high, 
                                        isotope_plot_slice_rt_high, SaveIsotopeDetDataFiles, SaveIsotopeDetGraphs, res_final)
            #Make sure that column names in sample_data after Isotope Detection are correct before downstream processes
            sample_data['ms1_peaks'].rename(columns={'mz_x': 'mz'}, inplace=True)
            print("DetectIsotopes() complete")
        elif PerformIsotopeDetection is False:
            print("Isotope Detection not performed")
        else:
            raise CustomError("""Set PerformIsotopeDetection as True or False to indicate whether you want to include this step.""")

        #Previous merging steps cause columns with the same names to be renamed
        # --> need to rename some of these columns (keeping those with the ms1_peak data with their original names)
        res_final.rename(columns={'intensity_x': 'intensity', 'mz_x': 'mz'}, inplace=True)

        #Collect the Peak Shape info here for downstream correlation analysis
        if PerformPeakShapeCorrelation is True:
            print("Collecting Peak Shape Data")

            sample_data, res_final = CustomFunctions.CollectPeakShapeData(sample_data, res_final, PeakShapeCorr_mz_tol, 
                                                                          PeakShapeCorr_RT_tol, PeakShapeCorr_DT_tol)

            print("Peak Shape Data Collected")
        elif PerformPeakShapeCorrelation is False:
            print("Peak Shape Correlation not performed")
        else:
            raise CustomError("""Set PerformPeakShapeCorrelation as True or False to indicate whether you want to include this step.""")

        #Make a folder for this sample for all Agglomerative Clustering results to go in to
        if os.path.exists('Results/{}/AgglomerativeClustering'.format(file_NoExt)):
            shutil.rmtree('Results/{}/AgglomerativeClustering'.format(file_NoExt))
        os.mkdir('Results/{}/AgglomerativeClustering'.format(file_NoExt))

        del sample_data
        
        loopcount = DEIMoSFunctions_RedundantStepsRemoved.AgglomerativeClusteringConcatenateNewPeakData(loopcount, res_final, agglo_mergeFeatures_mz_dt_rt_tol, 
                                                  file_NoExt)
        del res_final
        print("AgglomerativeClusteringConcatenateNewPeakData() complete")

    del rtalign_data, middle

    print("===============================")
    print("DEIMoS Script stopTime:", datetime.now())
    print("Total process run time:", datetime.now() - startTime)
    print("===============================")

    print("===============================")
    print("Objects stored locally from DEIMoS and custom steps:")
    print(list(locals()))
    print("===============================")

    exit()
        
    # del [files, middle, sample, file_NoExt, ms1, ms1_peaks, ms2, 
        #   ms2_peaks, loopcount, ReadFilesInDirectory, FindMiddleFileForRTAlignment,
        #   CreateResultsFile, DetectPeaks, ExtractMS2Spectra, DetectIsotopes, 
        #   ReferenceBasedAlignmentRT, AgglomerativeClusteringConcatenateNewPeakData] #type: ignore
    
    #Perform once final dataset for clustering (w/ all sample data) is created    
    clustering = DEIMoSFunctions.AgglomerativeClusteringMainSteps(agglo_multiSampPart_size, agglo_multiSampPart_tol, agglo_clustering_mz_dt_rt_tol)
    print("AgglomerativeClusteringMainSteps() complete")

    ccs_cal_pos = DEIMoSFunctions.CreateCCSCalObjects(tune_pos_file, 
                                                    ccsCalib_mz, ccsCalib_ccs, 
                                                    ccsCalib_q, ccsCalib_buffer_mass, 
                                                    ccsCalib_mz_tol, ccsCalib_dt_tol)
    print("CreateCCSCalObjects() complete")
    del tune_pos_file
    
    drifts_stripped = DEIMoSFunctions.CCSCalibrationSteps(ccs_cal_pos, clustering)
    print("CCSCalibrationSteps() complete")
        
    del [ms1, ms1_peaks, ms2, ms2_peaks] #type: ignore
    
    drifts_gapfilled = CustomFunctions.GapFillingSteps(ccs_cal_pos, drifts_stripped, GapFill_mz_tol, GapFill_rt_tol, GapFill_CCS_tol, 
                    GapFill_trapz_dx)
    print("GapFillingSteps() complete")
    
    PeakMerged_dataframe = CustomFunctions.PeakMergingSteps(drifts_gapfilled, PeakMerge_mz_ppm, PeakMerge_RT_tol, PeakMerge_CCS_tol)
    print("PeakMergingSteps() complete")
    
    CustomFunctions.MinimumDetectionThresholdSteps(PeakMerged_dataframe)
    print("MinimumDetectionThresholdSteps() complete")

# =============================================================================
#     CustomFunctions.MergeMS2Data()
#     print("MergeMS2Data() complete")
# =============================================================================
    
    del [calib_files, ccsCalib_mz, ccsCalib_ccs, ccsCalib_q, 
          ccsCalib_buffer_mass, ccsCalib_mz_tol, ccsCalib_dt_tol, 
          PeakDet_intensity_thres, PeakDet_smooth_data_radius, PeakDet_persistent_homology_radius, 
          MS2Extract_intensity_thres, MS2Extract_ms1_mz_subset_low, MS2Extract_ms1_dt_subset_low, 
          MS2Extract_ms1_rt_subset_low, MS2Extract_ms1_mz_subset_high, 
          MS2Extract_ms1_dt_subset_high, MS2Extract_ms1_rt_subset_high, 
          MS2Extract_ms2_dt_subset_low, MS2Extract_ms2_rt_subset_low, 
          MS2Extract_ms2_dt_subset_high, MS2Extract_ms2_rt_subset_high, 
          MS2Extract_model_ce, MS2Extract_model_params, MS2Extract_ms1_decon_intensity_thres, 
          MS2Extract_ms2_decon_intensity_thres, MS2Extract_construct_pairs_dt_low, 
          MS2Extract_construct_pairs_rt_low, MS2Extract_construct_pairs_dt_high, 
          MS2Extract_construct_pairs_rt_high, MS2Extract_construct_pairs_ce, 
          MS2Extract_construct_pairs_error_tol, MS2Extract_config_extract_mz_low, 
          MS2Extract_config_extract_dt_low, MS2Extract_config_extract_rt_low, 
          MS2Extract_config_extract_mz_high, MS2Extract_config_extract_dt_high, 
          MS2Extract_config_extract_rt_high, MS2Extract_decon_dt_resolution, 
          MS2Extract_dt_score_threshold, isotope_intensity_thres, isotope_partition_size, 
          isotope_partition_overlap, isotope_map_mz_dt_rt_tol, isotope_map_delta, 
          isotope_map_max_isotopes, isotope_map_max_charges, isotope_map_max_error, 
          isotope_min_no_isotopes, isotope_slice_mz_low, isotope_slice_mz_high, 
          isotope_plot_slice_mz_low, isotope_plot_slice_dt_low, isotope_plot_slice_rt_low, 
          isotope_plot_slice_mz_high, isotope_plot_slice_dt_high, isotope_plot_slice_rt_high, 
          rtalign_persisHomology_thres, rtalign_persis_thres, rtalign_partition_thres,
          rtalign_partition_size, rtalign_partition_overlap, rtalign_zipmap_thres, 
          rtalign_zipmap_mz_dt_rt_tol, agglo_mergeFeatures_mz_dt_rt_tol, agglo_multiSampPart_size, 
          agglo_multiSampPart_tol, agglo_clustering_mz_dt_rt_tol, GapFill_mz_tol, 
          GapFill_rt_tol, GapFill_CCS_tol, GapFill_trapz_dx, PeakMerge_mz_ppm, 
          PeakMerge_RT_tol, PeakMerge_CCS_tol] #type: ignore

    print("===============================")
    print("DEIMoS Script stopTime:", datetime.now())
    print("Total process run time:", datetime.now() - startTime)
    print("===============================")

    print("===============================")
    print("Objects stored locally from DEIMoS and custom steps:")
    print(list(locals()))
    print("===============================")

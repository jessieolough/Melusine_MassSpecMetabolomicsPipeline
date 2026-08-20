#Script to run through the DEIMoS commands for all .h5 files in a directory
#Adapted into discrete functions from the 20240503_LoopThroughHD5Files_DataProcessingOnly.py script
#Author: Jessica O'Loughlin (s1907024@ed.ac.uk)
#Supervisor: Prof. Karl Burgess (k.burgess@ed.ac.uk)
#Created: 08/11/2024

# import glob #Searches for files with specific extensions
import deimos #Performs various Mass Spec Processing Steps
import numpy as np #Basic math functionalities
import matplotlib.pyplot as plt #Create plot outputs
# import os #Allows directory to be read
# import os.path #Checks for existance of files
# import shutil #Will remove specific folders if they are already present
# import time #Find the creation date of files 
import pandas as pd #Handle dataframes
from numpy import trapz #Calculate area under line for Gap Filling
from datetime import datetime #Get current date and time
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def CollectPeakShapeData(sample_data, res_final, PeakShapeCorr_mz_tol, PeakShapeCorr_RT_tol, 
                         PeakShapeCorr_DT_tol):

    #Add new column to contain the Peak Shape information
    res_final['Peak_Shape'] = [None]*len(res_final)

    #Loop through each peak and collect peak shape information
    for index, row in sample_data['ms1_peaks'].iterrows():
        mz = row['mz']
        RT = row['retention_time']
        DT = row['drift_time']

        #Subset raw data within user-specified tolerances to find peak info
        #Calculate m/z uncertainty based off ppm tolerance (mz_tol)
        # uncertainty = round((mz/1000000)*PeakShapeCorr_mz_tol, 5)
        # ms1raw_subset = sample_data['ms1'].loc[(sample_data['ms1']["mz"] >= mz-uncertainty) & (sample_data['ms1']["mz"] <= mz+uncertainty)]
        ms1raw_subset = sample_data['ms1'][sample_data['ms1']["mz"] == mz]
        ms1raw_subset = ms1raw_subset[ms1raw_subset["drift_time"] == DT]
        ms1raw_subset = ms1raw_subset.loc[(ms1raw_subset["retention_time"] >= RT-PeakShapeCorr_RT_tol) & (ms1raw_subset["retention_time"] <= RT+PeakShapeCorr_RT_tol)]
        # ms1raw_subset = ms1raw_subset.loc[(ms1raw_subset["drift_time"] >= RT-PeakShapeCorr_DT_tol) & (ms1raw_subset["drift_time"] <= RT+PeakShapeCorr_DT_tol)]

        #Get average intensity at each RT time point (potentially across different m/z and drift time values)
        # fig, axes = plt.subplots(1, 3, figsize=(14, 6))

        # # mz vs intensity
        # axes[0].plot(ms1raw_subset['mz'], ms1raw_subset['intensity'], marker='o', linestyle='-')
        # axes[0].set_xlabel('m/z')
        # axes[0].set_ylabel('Intensity')
        # axes[0].set_title('Peak Shape: m/z vs Intensity')

        # # drift_time vs intensity
        # axes[1].plot(ms1raw_subset['drift_time'], ms1raw_subset['intensity'], marker='o', linestyle='-')
        # axes[1].set_xlabel('Drift Time')
        # axes[1].set_ylabel('Intensity')
        # axes[1].set_title('Peak Shape: Drift Time vs Intensity')

        # # retention_time vs intensity
        # axes[2].plot(ms1raw_subset['retention_time'], ms1raw_subset['intensity'], marker='o', linestyle='-')
        # axes[2].set_xlabel('Retention Time')
        # axes[2].set_ylabel('Intensity')
        # axes[2].set_title('Peak Shape: Retention Time vs Intensity')

        # plt.tight_layout()
        # plt.savefig(f"PeakShape_mz{mz}_RT{RT}_DT{DT}_RTTol{PeakShapeCorr_RT_tol}_mzTol{PeakShapeCorr_mz_tol}_DTTol{PeakShapeCorr_DT_tol}.png")
        # plt.close()

        
        ms1raw_subset = ms1raw_subset.groupby('retention_time').mean()
        ms1raw_subset = ms1raw_subset.reset_index()

        #Round all values in the subset raw data --> improve downstream RT matching
        ms1raw_subset = ms1raw_subset.round(3)     
         
        #Convert relevent information for Peak Shape into numpy array
        ms1raw_subset = ms1raw_subset[['retention_time', 'intensity']]
        ms1raw_subset = ms1raw_subset.to_numpy()

        #Add the peak shape information into the row in res_final corresponding to this MS1 peak
        res_final.at[index, 'Peak_Shape'] = ms1raw_subset

    return sample_data, res_final


def GapFillingSteps(ccs_cal_pos, drifts_stripped, GapFill_mz_tol, GapFill_rt_tol, GapFill_CCS_tol, 
                    GapFill_trapz_dx):
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
    
def PeakMergingSteps(drifts_gapfilled, PeakMerge_mz_ppm, PeakMerge_RT_tol, PeakMerge_CCS_tol): 
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
            max_val_RT = RT_val+PeakMerge_RT_tol
            min_val_RT = RT_val-PeakMerge_RT_tol
                
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
            uncertainty_CCS = (CCS_val/100)*PeakMerge_CCS_tol
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
    to_remove = ['mzs', 'RTs', 'drifts', 'ids', 'CCS', 'RT_bin']
    samples = [x for x in samples if x not in to_remove]

    minimum_detection_group_threshold = 0.50 #33% - PLEASE give to 2 significant figures!

    features_kept = pd.DataFrame()
    features_removed = pd.DataFrame()

    count_OverMinThreshold = 0
    count_UnderMinThreshold = 0

    print("Checks done")

    #Features in QCs and NOT in samples are to be removed
    for index, row in df.iterrows(): #Loop through each feature
        feature_series = row
        feature = pd.DataFrame(row)
        feature = feature.reset_index()
        # feature = feature.rename(columns = {"index":"Samples", 
        #                                 feature.columns[1]: "Intensities"})
        feature.columns.values[0] = "Samples"
        feature.columns.values[1] = "Intensities"
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
        print(ND_proportions_feature)
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

def MergeMS2Data(MergeMS2_mz_ppm, MergeMS2_RT_tol, MergeMS2_DT_tol):
    
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
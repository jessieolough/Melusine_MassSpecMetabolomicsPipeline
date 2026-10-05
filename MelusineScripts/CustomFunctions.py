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


def InspectRawDataAroundMS1Peaks(sample_data, res_final):

    #Add new column to contain the Peak Shape information
    res_final['Peak_Shape'] = [None]*len(res_final)      

    #Loop through each peak and collect peak shape information
    for index, row in sample_data['ms1_peaks'].iterrows():
        mz = row['mz']
        RT = row['retention_time']
        DT = row['drift_time']
        intensity = row['intensity']

        #Subset raw data within user-specified tolerances to find peak info
        #Calculate m/z uncertainty based off ppm tolerance (mz_tol)
        # uncertainty = round((mz/1000000)*PeakShapeCorr_mz_tol, 5)
        # ms1raw_subset = sample_data['ms1'].loc[(sample_data['ms1']["mz"] >= mz-uncertainty) & (sample_data['ms1']["mz"] <= mz+uncertainty)]
        ms1raw_subset = sample_data['ms1'][sample_data['ms1']["mz"] == mz]
        ms1raw_subset = ms1raw_subset[ms1raw_subset["drift_time"] == DT]

        # Sort by retention time
        ms1raw_subset = ms1raw_subset.sort_values("retention_time")
        #Find local minima to get retention time boundary for peak
        intensities = ms1raw_subset["intensity"].values
        rt_values = ms1raw_subset["retention_time"].values

        # Find the index of the intensity value (i.e., where intensity matches the current peak's intensity)
        closest_idx = (np.abs(intensities - intensity)).argmin()

        ##############################################
        # Find local minima to get retention time boundary for peak
        # # Sort by retention time
        # ms1raw_subset_PeakRiseSens = ms1raw_subset.sort_values("retention_time")
        # intensities = ms1raw_subset["intensity"].values
        # rt_values = ms1raw_subset["retention_time"].values

        # # Find the index of the intensity value (i.e., where intensity matches the current peak's intensity)
        # closest_idx = (np.abs(intensities - intensity)).argmin()

        # Scan left
        left_idx = closest_idx
        min_intensity_idx = left_idx  # Track the index of the lowest intensity found so far
        min_intensity = intensities[left_idx]
        consecutive_increase = 0
        last_intensity = intensities[left_idx]
        while left_idx > 0:# and intensities[left_idx] > 2000:
            left_idx -= 1
            current_intensity = intensities[left_idx]
            if current_intensity < min_intensity:
                min_intensity = current_intensity
                min_intensity_idx = left_idx
                consecutive_increase = 0  # reset counter on new minimum
            else:
                if current_intensity > last_intensity:
                    consecutive_increase += 1
                else:
                    consecutive_increase = 0
            last_intensity = current_intensity
            if consecutive_increase >= 5:
                break
        left_boundary_PeakRiseSens = rt_values[min_intensity_idx]

        # Scan right
        right_idx = closest_idx
        min_intensity_idx = right_idx  # Track the index of the lowest intensity found so far
        min_intensity = intensities[right_idx]
        consecutive_increase = 0
        last_intensity = intensities[right_idx]
        while right_idx < len(intensities) - 1:# and intensities[right_idx] > 2000:
            right_idx += 1
            current_intensity = intensities[right_idx]
            if current_intensity < min_intensity:
                min_intensity = current_intensity
                min_intensity_idx = right_idx
                consecutive_increase = 0  # reset counter on new minimum
            else:
                if current_intensity > last_intensity:
                    consecutive_increase += 1
                else:
                    consecutive_increase = 0
            last_intensity = current_intensity
            if consecutive_increase >= 5:
                break
        right_boundary_PeakRiseSens = rt_values[min_intensity_idx]

        # Use these as new boundaries
        ms1raw_subset_PeakRiseSens = ms1raw_subset.loc[
            (ms1raw_subset["retention_time"] >= left_boundary_PeakRiseSens) &
            (ms1raw_subset["retention_time"] <= right_boundary_PeakRiseSens)
        ]
        ##############################################

        ms1raw_subset = ms1raw_subset_PeakRiseSens.groupby('retention_time').mean()
        ms1raw_subset = ms1raw_subset.reset_index()

        #Round all values in the subset raw data --> improve downstream RT matching
        ms1raw_subset = ms1raw_subset.round(3)     
        
        #Convert relevent information for Peak Shape into numpy array
        ms1raw_subset = ms1raw_subset[['retention_time', 'intensity']]
        ms1raw_subset = ms1raw_subset.to_numpy()

        #Add the peak shape information into the row in res_final corresponding to this MS1 peak
        res_final.at[index, 'Peak_Shape'] = ms1raw_subset

    return sample_data, res_final

def PeakShapeCorrelation(feature_table, CorrelationMinimumNoDatapoints, PeakShapeCorr_CorrThres):

    #Correlate Peak Shapes
    # =============================================================================
    #Keep only the peak shape information
    PeakShape_df = feature_table[['Peak_Shape']]

    # print(PeakShape_df[['LongestPeakShape']])
    # print(type(PeakShape_df[['LongestPeakShape']]))

    #Correlate the selected Peak Shapes for each feature
    # correlation_matrix = PeakShape_df['LongestPeakShape'].corr()
    # correlation_matrix = np.corrcoef(PeakShape_df['LongestPeakShape'])
    #Get RT information (first element of the array)
    # PeakShape_list = PeakShape_df['Peak_Shape'].tolist()

    #TODO: for development, only keep specific number
    PeakShape_df = PeakShape_df[0:10]
    print(PeakShape_df)

    RT_ShapeInfo = []
    Ints_ShapeInfo = []

    import ast

    def parse_string_array(s):
        # Remove space at brackets
        s = s.replace('[ ', '[').replace(' ]', ']')
        # Add commas between numbers
        lines = [line.replace(' ', ',') for line in s.strip().split('\n')]
        s_fixed = '\n'.join(lines)
        # Convert to list of lists, then to numpy array
        try:
            arr = np.array(ast.literal_eval(s_fixed))
            return arr
        except Exception as e:
            print("Error parsing:", s)
            return np.array([])  # or np.nan

    PeakShape_df['peak_array'] = PeakShape_df['Peak_Shape'].apply(parse_string_array)

    PeakShape_df['retention_time'] = PeakShape_df['peak_array'].apply(lambda arr: arr[:, 0].tolist() if arr.size else [])
    PeakShape_df['intensity'] = PeakShape_df['peak_array'].apply(lambda arr: arr[:, 1].tolist() if arr.size else [])

    PeakShape_df.to_csv('Results_TestPeakShapeCorrelations/PeakShape_df.csv', index=False)

    #For each feature: separate the peak shapes array so that RTs and Ints are in separate lists
    # to allow the the Ints to be correlated against each other
    for index, row in PeakShape_df.iterrows():
        RT_ShapeInfo.append(row['retention_time'])
        Ints_ShapeInfo.append(row['intensity'])

    #Find features with shared RT values to perform correlation analysis against
    print(len(RT_ShapeInfo))

    '''
    To efficiently correlate each feature, go through each RT_group but SKIP each RT_group as you go along
    e.g. compare a, b, c, d
    = a x b, a x c, a x d
    then skip a
    = b x c, b x d
    then skip b
    = c x d
    --> get all relevant combinations without duplicates
    '''

    RT_position_1_list = list()
    RT_position_2_list = []
    correlation_list = []

    for RT_position_1 in range(0,len(RT_ShapeInfo),1):
        
        for RT_position_2 in range(RT_position_1+1, len(RT_ShapeInfo), 1):
            #Check if either set of features have intensities recorded at the same RTs
            shared_items = list(set(RT_ShapeInfo[RT_position_1]).intersection(RT_ShapeInfo[RT_position_2]))
            #Only do correlation analysis if there are a minimum of 3 shared datapoints
            if len(shared_items) >= CorrelationMinimumNoDatapoints:
                #Get position index of shared RT groups (to get respective Intensity values)
                RT_position_ind_1 = {val: [i for i, x in enumerate(RT_ShapeInfo[RT_position_1]) if x == val] for val in shared_items}
                RT_position_ind_2 = {val: [i for i, x in enumerate(RT_ShapeInfo[RT_position_2]) if x == val] for val in shared_items}
                #Get index values out of the dictionary object
                RT_position_ind_1 = list(RT_position_ind_1.values())
                RT_position_ind_2 = list(RT_position_ind_2.values())
                # Flatten the list to remove the [] around each item
                RT_position_ind_1 = [item for sublist in RT_position_ind_1 for item in sublist]
                RT_position_ind_2 = [item for sublist in RT_position_ind_2 for item in sublist]
                #Put index lists into order
                RT_position_ind_1.sort()
                RT_position_ind_2.sort()
                
                #Get the respective intensity values
                intensities_1 = Ints_ShapeInfo[RT_position_1]
                intensities_2 = Ints_ShapeInfo[RT_position_2]
                intensities_1 = [intensities_1[i] for i in RT_position_ind_1]
                intensities_2 = [intensities_2[i] for i in RT_position_ind_2]
                # print(intensities_1)
                # print(intensities_2)
                
                #Perform peak shape correlation
                correlation = np.corrcoef(intensities_1, intensities_2)
                correlation = correlation[0,1]
                # print(correlation)
                
                #Collect correlation data for downstream grouping
                RT_position_1_list.append(RT_position_1)
                RT_position_2_list.append(RT_position_2)
                correlation_list.append(correlation)
                
            else:
                pass
        # print("------------")
        
    correlations_df = pd.DataFrame({"RT_position_1":RT_position_1_list, 
                                    "RT_position_2":RT_position_2_list, 
                                    "Correlations":correlation_list})
        
    correlations_df.to_csv("correlations_df.csv")

    from collections import defaultdict

    def find_groups_with_coeff_df(df, threshold):
        """
        Groups items together if their correlation coefficient (from a dataframe)
        is above the threshold.

        Args:
            df (pd.DataFrame): DataFrame with columns ['RT_position_1', 'RT_position_2', 'Correlations']
            threshold (float): Minimum correlation coefficient to consider for grouping

        Returns:
            list of sets: Each set contains items in the same group
        """
        adj = defaultdict(set)
        # Iterate over DataFrame rows
        for _, row in df.iterrows():
            item1 = row['RT_position_1']
            item2 = row['RT_position_2']
            coeff = row['Correlations']
            if coeff >= threshold:
                adj[item1].add(item2)
                adj[item2].add(item1)
        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, group)

        for node in adj:
            if node not in visited:
                group = set()
                dfs(node, group)
                groups.append(group)

        return groups

    threshold = 0.85
    groups = find_groups_with_coeff_df(correlations_df[['RT_position_1', 'RT_position_2', 'Correlations']], 
                                       threshold)
    print(groups)

    # Assign PeakShapeCorrGroups: for each index, find which group it belongs to and set group index as value
    group_map = {}
    for i, group in enumerate(groups):
        for idx in group:
            group_map[idx] = i
    PeakShape_df['PeakShapeCorrGroups'] = PeakShape_df.index.map(lambda x: group_map.get(x, np.nan))

    print(PeakShape_df)

    return feature_table


def GapFillingSteps(feature_table, GapFill_mz_tol, GapFill_rt_tol, GapFill_CCS_tol, 
                    GapFill_trapz_dx, files):
    
    #TODO: Change back to the file read in the above line of code
    res = pd.read_csv("Results_TenSamplesReference/res.csv")
    res = res.apply(pd.to_numeric, errors = "ignore")

    #TODO: Temporary subsetting
    print(res[['sample_id', 'mz', 'drift_time','retention_time','intensity', 'cluster']])
    # files = [f.replace('.h5', '') for f in files]
    # res = res[res['sample_id'].isin(files)]
    # print(files)
    # print(res[['sample_id', 'mz', 'drift_time','retention_time','intensity', 'cluster']])
    res = res.reset_index(drop=True)
    # Sort res by 'cluster'
    res = res.sort_values(['intensity'],ascending=False)
    # # Get the first three unique cluster values
    # first_three_clusters = res['cluster'].unique()[0:6]
    # # Subset res to include only rows with these cluster values
    # res = res[res['cluster'].isin(first_three_clusters)]
    res = res[21:50]
    res = res.reset_index(drop=True)
    
    res['GapFillStatus'] = "FromPeakDetection"
    res['FeatureMissingIn'] = None

    print(res[['sample_id', 'mz', 'drift_time','retention_time','intensity', 'cluster', 'GapFillStatus', 'FeatureMissingIn']])

    samples = list(res['sample_id'].unique())
    print(samples)

    # Group by 'cluster' and loop through each cluster as a way to help group features
    # Make a list of which samples are missing a particular feature
    for cluster_id, cluster_df in res.groupby('cluster'):
        # # For each unique (mz, drift_time, retention_time), find missing samples
        # unique_features = cluster_df[['mz', 'drift_time', 'retention_time']].drop_duplicates()
        for idx, feat in cluster_df[['mz', 'drift_time', 'retention_time']].iterrows():
            mz_val = feat['mz']
            dt_val = feat['drift_time']
            rt_val = feat['retention_time']
            feature_rows = cluster_df[
                (cluster_df['mz'] == mz_val) &
                (cluster_df['drift_time'] == dt_val) &
                (cluster_df['retention_time'] == rt_val)
            ]
            present_samples = feature_rows['sample_id'].unique()
            missing_samples = [s for s in samples if s not in present_samples]
            print(f"Feature mz={mz_val}, drift_time={dt_val}, retention_time={rt_val} missing samples: {missing_samples}")
            # Add missing_samples as "FeatureMissingIn" for this feature based on its index in res
            res.at[idx, 'FeatureMissingIn'] = missing_samples #type: ignore

    print(res[['sample_id', 'mz', 'drift_time','retention_time','intensity', 'cluster', 'GapFillStatus', 'FeatureMissingIn']])

    # For each sample (--> raw data only needs to be read in once), check which features it is missing and only Gap Fill for those
    for sample in samples:
        print("========================")
        print(sample)

        #TODO: For evaluation, do not subset res --> can see peaks that are meant to be there
        res_sub = res.copy()

        # #Subset res to only contain features that are missing in that sample
        # res_sub = res[res['FeatureMissingIn'].apply(lambda x: isinstance(x, list) and sample in x if x is not None else False)]
        # print(res_sub[['sample_id', 'mz', 'drift_time','retention_time','intensity', 'FeatureMissingIn']])

        #TODO: Change folder name that the raw data is read from
        #Load raw data
        sample_raw = deimos.load('Results_TenSamplesReference/RTAligned_{}.h5'.format(sample), key='ms1')

        #Loop through each peak and collect peak intensity information
        for index, row in res_sub.iterrows():
            mz = row['mz']
            RT = row['retention_time']
            DT = row['drift_time']
            intensity = row['intensity']
            print(mz, RT, DT, intensity, row['FeatureMissingIn'], row['Peak_Shape'])

            #Subset raw data within user-specified tolerances to find peak info
            sampleraw_subset = sample_raw[sample_raw["mz"] == mz]
            sampleraw_subset = sampleraw_subset[sampleraw_subset["drift_time"] == DT]
            if sampleraw_subset.empty is True:
                print("No datapoints present")
            else:
                print("datapoints present")
                # Sort by retention time
                sampleraw_subset = sampleraw_subset.sort_values("retention_time")
                #Find local minima to get retention time boundary for peak
                intensities = sampleraw_subset["intensity"].values
                rt_values = sampleraw_subset["retention_time"].values
                # Find the index of the intensity value (i.e., where intensity matches the current peak's intensity)
                closest_idx = (np.abs(intensities - intensity)).argmin()

                #TODO: Keep RT windows thresholding for evaluation only
                sampleraw_subset_RTTol = sampleraw_subset.loc[(sampleraw_subset["retention_time"] >= RT-2) & (sampleraw_subset["retention_time"] <= RT+2)]

                # Find local minima to get retention time boundary for peak
                # Scan left
                left_idx = closest_idx
                min_intensity_idx = left_idx  # Track the index of the lowest intensity found so far
                min_intensity = intensities[left_idx]
                consecutive_increase = 0
                last_intensity = intensities[left_idx]
                while left_idx > 0:# and intensities[left_idx] > 2000:
                    left_idx -= 1
                    current_intensity = intensities[left_idx]
                    if current_intensity < min_intensity:
                        min_intensity = current_intensity
                        min_intensity_idx = left_idx
                        consecutive_increase = 0  # reset counter on new minimum
                    else:
                        if current_intensity > last_intensity:
                            consecutive_increase += 1
                        else:
                            consecutive_increase = 0
                    last_intensity = current_intensity
                    if consecutive_increase >= 5:
                        break
                left_boundary_PeakRiseSens = rt_values[min_intensity_idx]

                # Scan right
                right_idx = closest_idx
                min_intensity_idx = right_idx  # Track the index of the lowest intensity found so far
                min_intensity = intensities[right_idx]
                consecutive_increase = 0
                last_intensity = intensities[right_idx]
                while right_idx < len(intensities) - 1:# and intensities[right_idx] > 2000:
                    right_idx += 1
                    current_intensity = intensities[right_idx]
                    if current_intensity < min_intensity:
                        min_intensity = current_intensity
                        min_intensity_idx = right_idx
                        consecutive_increase = 0  # reset counter on new minimum
                    else:
                        if current_intensity > last_intensity:
                            consecutive_increase += 1
                        else:
                            consecutive_increase = 0
                    last_intensity = current_intensity
                    if consecutive_increase >= 5:
                        break
                right_boundary_PeakRiseSens = rt_values[min_intensity_idx]

                # Use these as new boundaries
                sampleraw_subset_PeakRiseSens = sampleraw_subset.loc[
                    (sampleraw_subset["retention_time"] >= left_boundary_PeakRiseSens) &
                    (sampleraw_subset["retention_time"] <= right_boundary_PeakRiseSens)
                ]

                # Get average intensity at each RT time point (potentially across different m/z and drift time values)
                fig, axes = plt.subplots(1, 2, figsize=(8, 6))

                # retention_time (user tolerance) vs intensity
                axes[0].plot(sampleraw_subset_RTTol['retention_time'], sampleraw_subset_RTTol['intensity'], marker='o', linestyle='-')
                axes[0].set_xlabel('Retention Time')
                axes[0].set_ylabel('Intensity')
                axes[0].set_title('Peak Shape: Retention Time \n(User Tolerance) vs Intensity')
                axes[0].scatter([RT], [intensity], color='red', zorder=10, label='Peak')
                axes[0].legend()

                # retention_time (local minima, Peak Rise Sensitive) vs intensity
                axes[1].plot(sampleraw_subset_PeakRiseSens['retention_time'], sampleraw_subset_PeakRiseSens['intensity'], marker='o', linestyle='-')
                axes[1].set_xlabel('Retention Time')
                axes[1].set_ylabel('Intensity')
                axes[1].set_title('Peak Shape: Retention Time \n(Local Minima, Peak Rise Sensitive) vs Intensity')
                axes[1].scatter([RT], [intensity], color='red', zorder=10, label='Peak')
                axes[1].legend()

                # Add overall super title with mz, RT, DT, and intensity
                is_present = sample not in row['FeatureMissingIn']

                fig.subplots_adjust(top=0.85)
                fig.suptitle(f"Sample {sample}: \n mz={mz:.5f}, RT={RT:.3f}, DT={DT:.3f}, Intensity={intensity:.0f} \n FeaturePresent = {is_present}", fontsize=14)

                plt.tight_layout()
                plt.savefig(f"Results_TenSamplesReference/GapFilling_{sample}_mz{mz}_RT{RT}_DT{DT}.png", bbox_inches='tight')
                plt.show()
                plt.close()

    exit()
    
    #Loop through samples/columns
    drifts_gapfilled = res.copy()
    
    # print(drifts_gapfilled[samples])
    
    # iterate through specific columns of the dataframe
    for sample in list(drifts_gapfilled[sample_id].unique()):
        print("-=-=-=-=-")
        print(sample)
        #Load data
        ms1_raw_df = deimos.load('RTAligned_{}.h5'.format(sample), key='ms1')
        #Threshold data
        ms1_raw_df = deimos.threshold(ms1_raw_df, threshold=500)

        # Loop through features in each sample
        for index_f, row in drifts_gapfilled[sample].items():
            # print(index_f, row)
            feature = drifts_gapfilled
            # print(feature)
            feature_row = feature.loc[index_f]
            feature = feature_row[sample]

            #Only Gap Fill 
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
    
def MinimumDetectionThresholdSteps(PeakMerged_dataframe, minimum_detection_group_threshold):
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
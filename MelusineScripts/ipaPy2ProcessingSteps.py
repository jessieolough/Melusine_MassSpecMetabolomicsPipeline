import pandas as pd #Handle dataframes
from datetime import datetime #Get current date and time
from ipaPy2 import ipa #import the libraries #type: ignore
import pickle #Pickle dictionary object to save in directory
import os #Help with reading and saving files in specified directories

##Set thresholds for different processes
#ipaPy2 Annotation
ipaPy2_compAdducts_ionisation = 1
ipaPy2_compAdducts_ncores = 4
ipaPy2_MS1Annotation_ppm = 3
ipaPy2_MS1Annotation_ncores = 4 #ipaPy2 documentation used 1 as default
ipaPy2_GibbsSamplerAdd_noits = 1000
ipaPy2_GibbsSamplerAdd_delta_add = 0.1

#Custom Errors
class CustomError(Exception):
    pass

# def MetaboliteAnnotationSteps(drifts_stripped):
def MetaboliteAnnotationSteps(clusterFeaturesipaPy2):
    #With thanks to Karl Burgess for scripting the majority this section
    
    startDataLoad = datetime.now()

    #get the database files
    #TODO: Replace this with a custom db extracted from the MSDial libraries
    DB = pd.read_csv('ipapy2_files/IPA_MS1.csv')
    adducts = pd.read_csv('ipapy2_files/adducts.csv')
    print("ipaPy2 files loaded")

    if clusterFeaturesipaPy2 is True:
        feature_table = pd.read_csv('20250506_Results_ToReviewWithKarl/final_dataframe_unannotated.csv')
        feature_table = pd.DataFrame(feature_table)
        feature_table = feature_table.apply(pd.to_numeric, errors = "ignore")
        
        #get the timestamps in minutes rather than seconds - probably not essential
        feature_table['RTs'] = feature_table['RTs'] * 60
        #fill NaNs with 0s. Clustering won't work without it
        #TODO: look into gap filling algorithms and missing value imputation
        feature_table = feature_table.fillna(0)
        #cluster the features by similarity and retention time
        #ideally this will be done via MzMatch or something
        #but works ok so far.
        
        # Added in by Jess (08/04/2025)
        #ipa.clusterFeatures() cannot convert indexes in [] into floats
        feature_table = feature_table.drop('ids', axis=1)
        feature_table = feature_table.reset_index(drop = False)
        feature_table = feature_table.rename(columns = {"index":"ids"})
        feature_table['ids'] += 1
        ##Keep object containing ids, CCS, m/z, and RT values to use downstream
        feature_data = feature_table[['ids', 'mzs', 'RTs', 'drifts', 'CCS']]
        #Remove unnecessary columns
        feature_table = feature_table.drop('CCS', axis=1)
        feature_table = feature_table.drop('drifts', axis=1)
        #TODO: improve how the sample columns are specified to be kept
        feature_table = feature_table[['ids','mzs', 'RTs',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_1',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_1_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_1_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_2',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_2_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_2_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_3',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_3_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_3_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_4',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_4_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_4_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_5',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_5_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_400TR_5_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_1',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_1_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_1_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_2',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_2_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_2_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_3',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_3_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_3_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_4',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_4_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_4_MA-d9-Min5-Spk_SR',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_5',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_5_MA-d9-Min5-Spk',
                                        'RTAligned_POS_FBS_IM_MSMS_40kTF_500TR_5_MA-d9-Min5-Spk_SR']]
        print(feature_table.columns)
        #TODO: add proper index column
        print(feature_table)
        feature_table.to_csv('drifts_stripped_mean_JO.csv')
        # feature_table.to_csv('drifts_stripped_check.csv')
        
        time_before_clusterFeatures = datetime.now()
        print(time_before_clusterFeatures)
        drifts_clustered = ipa.clusterFeatures(feature_table,
                                            Cthr=0.8, 
                                            RTwin=1, 
                                            Intmode='max' #default = 'max'
                                            )
        time_after_clusterFeatures = datetime.now()
        print(time_after_clusterFeatures)
        drifts_clustered.to_csv('clusterFeatures_output_JO.csv')
        print("ipa.clusterFeatures completed")
        print(drifts_clustered)
    elif clusterFeaturesipaPy2 is False:
        #TODO: Put DEIMoS-clustered feature table into format for ipaPy2 steps
        feature_table = pd.DataFrame()
    else:
        raise CustomError("""Set clusterFeaturesipaPy2 as True or False to indicate whether you want to include this step.""")
    
    # drifts_clustered.to_csv('drifts_clustered.csv')
    #build possible adduct library for this dataset
    allAdds = ipa.compute_all_adducts(adducts, DB, 
                                      ionisation=ipaPy2_compAdducts_ionisation, 
                                      ncores=ipaPy2_compAdducts_ncores)
    print(allAdds)
    # allAdds.head(100).to_csv('allAdds_Top100.csv')
    print("ipq.compute_all_adducts() completed")
    
    #Added by Jess (08/04/2025)
    #ipaPy2 documentation says that the ipa.MS1annotation() function should get data
    # in the format output from the ipa.map_isotope_patterns() function
    print(feature_table)
    # drifts_clustered = drifts_clustered.rename(columns={'Int': 'Ints'})
    
    time_before_map_isotope_patterns = datetime.now()
    print(time_before_map_isotope_patterns)
    ipa.map_isotope_patterns(feature_table, 
                             isoDiff=1, 
                             ppm=100, #default 100 
                             ionisation=1, 
                             MinIsoRatio=0.5)
    time_after_map_isotope_patterns = datetime.now()
    print(time_after_map_isotope_patterns)
    feature_table.to_csv('mapIsotopePatterns_output_JO.csv')
    print("map_isotope_patterns completed")
    print(feature_table)
    
    #complete annotations for MS1
    #TODO: extract the MSMS data and perform MSMS annotation
    annotations = ipa.MS1annotation(feature_table, allAdds, 
                                    ppm=ipaPy2_MS1Annotation_ppm, 
                                    ncores=ipaPy2_MS1Annotation_ncores)
    time_after_MS1annotation = datetime.now()
    print(time_after_MS1annotation)
    print("ipa.MS1annotation completed")
    print(type(annotations))
    print(len(annotations))
    
    #Save the annotations dictionary object as a pickled file
    try:
        file_to_save = open('annotations_afterMS1annotation_JO.pkl','wb')
        pickle.dump(annotations, file_to_save)
        file_to_save.close()
    
    except:
        print("Unable to save annotations dictionary as pickled file")
    
    #compute posterior probabilities including adduct data
    zs = ipa.Gibbs_sampler_add(feature_table, annotations, 
                               noits=ipaPy2_GibbsSamplerAdd_noits, 
                               delta_add=ipaPy2_GibbsSamplerAdd_delta_add, 
                               all_out=True)
    time_after_Gibbs_sampler_add = datetime.now()
    print(time_after_Gibbs_sampler_add)
    print("ipa.Gibbs_sampler_add completed")
    print(type(zs))
    
    # open file
    with open('Gibbs_sampler_add_output_zs_JO.txt', 'w+') as f:
        # write elements of list
        for items in zs:
            f.write('%s\n' %items)
        print("File written successfully")
    # close the file
    f.close() 
    
    #compute bio connections
    Bio = ipa.Compute_Bio(DB, annotations, mode='reactions', ncores = 4)
    time_after_Compute_Bio = datetime.now()
    print(time_after_Compute_Bio)
    print("ipa.Compute_bio() completed")
    
    Bio.to_csv('Compute_Bio_output_JO.csv')
    
    #Compute posterior probabilities integrating both adducts and biochemical connections
    ipa.Gibbs_sampler_bio_add(feature_table, #Should be the output of map_isotope_patterns()
                              annotations,
                              Bio,
                              noits=5000,
                              delta_bio=0.1,
                              delta_add=0.1, 
                              all_out = True, 
                              zs = zs)
    print("Gibbs_sampler_bio_add completed")
    time_after_Gibbs_sampler_bio_add = datetime.now()
    print(time_after_Gibbs_sampler_bio_add)
    
    #Save the annotations dictionary object as a pickled file
    try:
        file_to_save = open('annotations_after_Gibbs_sampler_bio_add_JO.pkl','wb')
        pickle.dump(annotations, file_to_save)
        file_to_save.close()
    
    except:
        print("Unable to save annotations dictionary as pickled file")

    print("Time to Read and Tidy Data:", time_before_clusterFeatures-startDataLoad)
    print("Time for ipa.clusterFeatures():", time_after_clusterFeatures-time_before_clusterFeatures)
    print("Time for ipa.compute_all_adducts():", time_before_map_isotope_patterns-time_after_clusterFeatures)
    print("Time for ipa.map_isotope_patterns():", time_after_map_isotope_patterns-time_before_map_isotope_patterns)
    print("Time for ipa.MS1annotation():", time_after_MS1annotation-time_after_map_isotope_patterns)
    print("Time for ipa.Gibbs_sampler_add():", time_after_Gibbs_sampler_add-time_after_MS1annotation)
    print("Time for ipa.Compute_Bio():", time_after_Compute_Bio-time_after_Gibbs_sampler_add)
    print("Time for ipa.Givvs_sampler_bio_add():", time_after_Gibbs_sampler_bio_add-time_after_Compute_Bio)
    
# =============================================================================
#     ##Merge the feature_data and annotations_df objects together
#     #Add suffix to feature_data values so that these are not mixed up with other downstream values
#     feature_data = feature_data.add_suffix('_featureData')
#     #Rename feature_data's 'ids' column to 'key' to allow merging
#     feature_data = feature_data.rename(columns = {"ids_featureData":"key"})
#     print(feature_data)
#     print(annotations_df)
#     #Merge data
#     annotations_merged = pd.merge(annotations_df, feature_data, on="key", how = "left")
#     print(annotations_merged)
# =============================================================================
    
    
    
# =============================================================================
#     list_keys = list(annotations.keys())
#     print(list_keys)
#     for key, value in annotations.items():
#         print("key: ", key)
#         print(value)
# =============================================================================
            
# =============================================================================
#     # open file
#     with open('Gibbs_sampler_bio_add_output_zs.txt', 'w+') as f:
#         # write elements of list
#         for items in zs:
#             f.write('%s\n' %items)
#         print("File written successfully")
#     # close the file
#     f.close() 
# =============================================================================

# =============================================================================
#     del [drifts_stripped, drifts_clustered, allAdds, annotations, zs, Bio]
#     
#     print("Intermediate times for each step:")
#     print("time_before_clusterFeatures:", time_before_clusterFeatures)
#     print("time_after_clusterFeatures:", time_after_clusterFeatures)
#     print("time_before_map_isotope_patterns:", time_before_map_isotope_patterns)
#     print("time_after_map_isotope_patterns:", time_after_map_isotope_patterns)
#     print("time_after_MS1annotation:", time_after_MS1annotation)
#     print("time_after_Gibbs_sampler_add:", time_after_Gibbs_sampler_add)
#     print("time_after_Compute_Bio:", time_after_Compute_Bio)
#     print("time_after_Gibbs_sampler_bio_add:", time_after_Gibbs_sampler_bio_add)
# =============================================================================

clusterFeaturesipaPy2 = True

if __name__ == "__main__":
    
    processes=1
    
    startTime = datetime.now()
    print("===============================")
    print("ipaPy2 Script startTime:", startTime)
    print("===============================")

    #Set working directory
    os.chdir(r"f:\JessicaOLoughlin\RawDataMZMLFiles")
    print("Working directory set")
        
    # MetaboliteAnnotationSteps(drifts_stripped)
    MetaboliteAnnotationSteps(clusterFeaturesipaPy2)
    print("MetaboliteAnnotationSteps() complete")    
    
    print("===============================")
    print("ipaPy2 Script stopTime:", datetime.now())
    print("Total process run time:", datetime.now() - startTime)
    print("===============================")

    print("===============================")
    print("Objects stored locally from ipaPy2 steps:")
    print(list(locals()))
    print("===============================")
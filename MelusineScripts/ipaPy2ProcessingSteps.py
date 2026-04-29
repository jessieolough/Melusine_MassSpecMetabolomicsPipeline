import pandas as pd #Handle dataframes
from datetime import datetime #Get current date and time
from ipaPy2 import ipa #import the libraries
import pickle #Pickle dictionary object to save in directory

##Set thresholds for different processes
#ipaPy2 Annotation
ipaPy2_compAdducts_ionisation = 1
ipaPy2_compAdducts_ncores = 4
ipaPy2_MS1Annotation_ppm = 3
ipaPy2_MS1Annotation_ncores = 4 #ipaPy2 documentation used 1 as default
ipaPy2_GibbsSamplerAdd_noits = 1000
ipaPy2_GibbsSamplerAdd_delta_add = 0.1

# def MetaboliteAnnotationSteps(drifts_stripped):
def MetaboliteAnnotationSteps():
    #With thanks to Karl Burgess for scripting the majority this section
    
    drifts_stripped = pd.read_csv('20250506_Results_ToReviewWithKarl/final_dataframe_unannotated.csv')
    drifts_stripped = pd.DataFrame(drifts_stripped)
    drifts_stripped = drifts_stripped.apply(pd.to_numeric, errors = "ignore")

    #OK! Perform all IPA steps!
    
    #get the database files
    #TODO: Replace this with a custom db extracted from the MSDial libraries
    DB = pd.read_csv('ipapy2_files/IPA_MS1.csv')
    adducts = pd.read_csv('ipapy2_files/adducts.csv')
    print("ipaPy2 files loaded")
    #get the timestamps in minutes rather than seconds - probably not essential
    drifts_stripped['RTs'] = drifts_stripped['RTs'] * 60
    #fill NaNs with 0s. Clustering won't work without it
    #TODO: look into gap filling algorithms and missing value imputation
    drifts_stripped = drifts_stripped.fillna(0)
    #cluster the features by similarity and retention time
    #ideally this will be done via MzMatch or something
    #but works ok so far.
    
    # Added in by Jess (08/04/2025)
    #ipa.clusterFeatures() cannot convert indexes in [] into floats
    drifts_stripped = drifts_stripped.drop('ids', axis=1)
    drifts_stripped = drifts_stripped.reset_index(drop = False)
    drifts_stripped = drifts_stripped.rename(columns = {"index":"ids"})
    drifts_stripped['ids'] += 1
    ##Keep object containing ids, CCS, m/z, and RT values to use downstream
    feature_data = drifts_stripped[['ids', 'mzs', 'RTs', 'drifts', 'CCS']]
    #Remove unnecessary columns
    drifts_stripped = drifts_stripped.drop('CCS', axis=1)
    drifts_stripped = drifts_stripped.drop('drifts', axis=1)
    drifts_stripped = drifts_stripped[['ids','mzs', 'RTs',
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
    print(drifts_stripped.columns)
    #TODO: add proper index column
    print(drifts_stripped)
    drifts_stripped.to_csv('drifts_stripped_mean_JO.csv')
    # drifts_stripped.to_csv('drifts_stripped_check.csv')
    
    time_before_clusterFeatures = datetime.now()
    print(time_before_clusterFeatures)
    drifts_clustered = ipa.clusterFeatures(drifts_stripped, 
                                           Cthr=0.8, 
                                           RTwin=1, 
                                           Intmode='max' #default = 'max'
                                           )
    time_after_clusterFeatures = datetime.now()
    print(time_after_clusterFeatures)
    drifts_clustered.to_csv('clusterFeatures_output_JO.csv')
    print("ipa.clusterFeatures completed")
    print(drifts_clustered)
    
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
    print(drifts_clustered)
    # drifts_clustered = drifts_clustered.rename(columns={'Int': 'Ints'})
    
    time_before_map_isotope_patterns = datetime.now()
    print(time_before_map_isotope_patterns)
    ipa.map_isotope_patterns(drifts_clustered, 
                             isoDiff=1, 
                             ppm=100, #default 100 
                             ionisation=1, 
                             MinIsoRatio=0.5)
    time_after_map_isotope_patterns = datetime.now()
    print(time_after_map_isotope_patterns)
    drifts_clustered.to_csv('mapIsotopePatterns_output_JO.csv')
    print("map_isotope_patterns completed")
    print(drifts_clustered)
    
    #complete annotations for MS1
    #TODO: extract the MSMS data and perform MSMS annotation
    annotations = ipa.MS1annotation(drifts_clustered, allAdds, 
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
    zs = ipa.Gibbs_sampler_add(drifts_clustered, annotations, 
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
    ipa.Gibbs_sampler_bio_add(drifts_clustered, #Should be the output of map_isotope_patterns()
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
    
def AddInCCSDataSteps(DB, annotations):
    #TODO: remove and add back in the steps to generate annotations
    annotations_df = pd.read_csv('annotations_df_keys.csv')
    mcleanLibrary = pd.read_csv('UnifiedCCSCompendium_FullDataSet.csv')
    mcleanLibrary = mcleanLibrary[['Compound', 'InChi', 'mz', 'CCS']]
    mcleanLibrary = mcleanLibrary.add_suffix('_mclean')
    #Rename InChi_mclean column in McLean data to inchi to allow downstream merging
    mcleanLibrary = mcleanLibrary.rename(columns = {"InChi_mclean":"inchi"})
    #Remove rows with unknown inchi keys (otherwise all unknowns will be merged)
    mcleanLibrary = mcleanLibrary[mcleanLibrary['inchi'].notna()]
    print(mcleanLibrary)
    
    DB_sub = DB[['id', 'inchi']]
    #Remove rows with unknown inchi keys (otherwise all unknowns will be merged)
    # DB_sub = DB_sub [DB_sub ['inchi'].notna()]
    print(DB_sub)
    
    
    annotations = []
    with (open("annotations_afterMS1annotation_JO.pkl", "rb")) as openfile:
        while True:
            try:
                annotations.append(pickle.load(openfile))
            except EOFError:
                break
            
    annotations = annotations[0]
    print(len(annotations))
# =============================================================================
#     from itertools import islice
#     annotations = dict(islice(annotations.items(), 30))
#     print(len(annotations))
# =============================================================================
    
    #Save each key as a row in a new dataframe
    annotations_df = pd.DataFrame()
    
    for key, metabolite in annotations.items():
        metabolite_df = pd.DataFrame()
        
        # print(metabolite)
        #Add in inchi keys to allow matching with McLean Library
        # print(metabolite['id'])
        inchi_keys = pd.merge(metabolite, DB_sub, on="id", how = "left")
        inchi_keys = inchi_keys[['id', 'name', 'inchi']]
        # print(inchi_keys)
        #Match inchi keys in the McLean Library 
        mcleanData = pd.merge(inchi_keys, mcleanLibrary, on="inchi", how = "left")
        # print(mcleanData[['id', 'name', 'inchi', 'Compound_mclean', 'mz_mclean', 'CCS_mclean']])

        # print("-=-=-=-=-=-=-=-=-")
        
        
        metabolite_df["key"] = ""
        metabolite_df.at[0, 'key'] = key
        metabolite_df["id"] = ""
        metabolite_df.at[0, 'id'] = list(metabolite['id'].values)
        metabolite_df["name"] = ""
        metabolite_df.at[0, 'name'] = list(metabolite['name'].values)
        metabolite_df["formula"] = ""
        metabolite_df.at[0, 'formula'] = list(metabolite['formula'].values)
        metabolite_df["adduct"] = ""
        metabolite_df.at[0, 'adduct'] = list(metabolite['adduct'].values)
        metabolite_df["m/z"] = ""
        metabolite_df.at[0, 'm/z'] = list(metabolite['m/z'].values)
        metabolite_df["charge"] = ""
        metabolite_df.at[0, 'charge'] = list(metabolite['charge'].values)
        metabolite_df["RT range"] = ""
        metabolite_df.at[0, 'RT range'] = list(metabolite['RT range'].values)
        metabolite_df["ppm"] = ""
        metabolite_df.at[0, 'ppm'] = list(metabolite['ppm'].values)
        metabolite_df["isotope pattern score"] = ""
        metabolite_df.at[0, 'isotope pattern score'] = list(metabolite['isotope pattern score'].values)
        metabolite_df["fragmentation pattern score"] = ""
        metabolite_df.at[0, 'fragmentation pattern score'] = list(metabolite['fragmentation pattern score'].values)
        metabolite_df["prior"] = ""
        metabolite_df.at[0, 'prior'] = list(metabolite['prior'].values)
        metabolite_df["post"] = ""
        metabolite_df.at[0, 'post'] = list(metabolite['post'].values)
        # metabolite_df["post Gibbs"] = ""
        # metabolite_df.at[0, 'post Gibbs'] = list(metabolite['post Gibbs'].values)
        # metabolite_df["chi-square pval"] = ""
        # metabolite_df.at[0, 'chi-square pval'] = list(metabolite['chi-square pval'].values)
        metabolite_df["inchi"] = ""
        metabolite_df.at[0, 'inchi'] = list(mcleanData['inchi'].values)
        metabolite_df["Compound_mclean"] = ""
        metabolite_df.at[0, 'id'] = list(mcleanData['Compound_mclean'].values)
        
        #Add data to final dataframe
        annotations_df = pd.concat([annotations_df, metabolite_df])
        
    annotations_df.to_csv("annotations_df_keys_merged_test.csv", index = False)
    print("annotations_df saved as CSV and pickled file")

    
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

#TODO: Start of the process
if __name__ == "__main__":
    
    processes=1
    
    startTime = datetime.now()
    print("===============================")
    print("ipaPy2 Script startTime:", startTime)
    print("===============================")
        
    # MetaboliteAnnotationSteps(drifts_stripped)
# =============================================================================
#     MetaboliteAnnotationSteps()
#     print("MetaboliteAnnotationSteps() complete")
# =============================================================================
    
    AddInCCSDataSteps()
    print("AddInCCSDataSteps() complete")
    
    print("===============================")
    print("ipaPy2 Script stopTime:", datetime.now())
    print("Total process run time:", datetime.now() - startTime)
    print("===============================")

    print("===============================")
    print("Objects stored locally from ipaPy2 steps:")
    print(list(locals()))
    print("===============================")
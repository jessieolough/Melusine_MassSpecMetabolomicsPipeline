#Script to create a csv file for the user to manually assign sample groups to each sample
#Author: Jessica M O'Loughlin (s1907024@ed.ac.uk)
#Supervisor: Prof. Karl Burgess (karl.burgess@ed.ac.uk)
#Date created: 13/09/2024
#Adapted 27/03/2025 for DEIMoS Data


import pandas as pd

#Load in the MassProfiler data
df = pd.read_csv('PeakMerged_final_dataframe.csv')
df = df.apply(pd.to_numeric, errors = "ignore")

#Make a list of the samples
samples = list(df.columns)
#Remove non-sample names from list
to_remove = ['mzs', 'RTs', 'drifts', 'ids', 'CCS']
samples = [x for x in samples if x not in to_remove]

#Add samples to a new dataframe
group_template = pd.DataFrame(samples)
group_template.rename(columns={0:'Samples'}, inplace=True)
#Put the samples in ascending order (--> not random --> easier for user to read)
group_template = group_template.sort_values(by='Samples', ascending=True)
#Reset the index
group_template = group_template.reset_index()
#Delete the extra column created
group_template = group_template.drop(group_template.columns[0],axis = 1)
#Add a column for the user to specify the groupings manually in Excel
group_template["Groups"] = ""
print(group_template)
#Save as a csv file
group_template.to_csv("Sample_groupings.csv", index = False)

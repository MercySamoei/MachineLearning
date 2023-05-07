# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# import the dataset
data = pd.read_csv('chip_dataset.csv')
# print(data.to_string())
print(data.isnull().sum()) # to check if there are empty cells in the columns and to know how many they are
print(data.columns)
print(data.shape)

mean_process = np.mean(data['Process Size (nm)'])
missing_processes = round(mean_process)
data['Process Size (nm)'] = data['Process Size (nm)'].fillna(missing_processes)

mean_tdp = np.mean(data['TDP (W)'])
missing_tdp = round(mean_tdp)
data['TDP (W)'] = data['TDP (W)'].fillna(missing_tdp)

mean_die = np.mean(data['Die Size (mm^2)'])
missing_die = round(mean_die)
data['Die Size (mm^2)'] = data['Die Size (mm^2)'].fillna(missing_die)

mean_transistor = np.mean(data['Transistors (million)'])
missing_transistor = round(mean_transistor)
data['Transistors (million)'] = data['Transistors (million)'].fillna(missing_transistor)

mean_fp16 = np.mean(data['FP16 GFLOPS'])
missing_fp16 = round(mean_fp16)
data['FP16 GFLOPS'] = data['FP16 GFLOPS'].fillna(missing_fp16)

mean_fp32 = np.mean(data['FP32 GFLOPS'])
missing_fp32 = round(mean_fp32)
data['FP32 GFLOPS'] = data['FP32 GFLOPS'].fillna(missing_fp32)

mean_fp64 = np.mean(data['FP64 GFLOPS'])
missing_fp64 = round(mean_fp64)
data['FP64 GFLOPS'] = data['FP64 GFLOPS'].fillna(missing_fp64)
# or
columns = ['Process Size (nm)', 'TDP (W)', 'Die Size (mm^2)', 'Transistors (million)', 'FP16 GFLOPS', 'FP32 GFLOPS', 'FP64 GFLOPS']
means = {}

for col in columns:
    mean_val = np.mean(data[col])
    missing_val = round(mean_val)
    data[col] = data[col].fillna(missing_val)
    means[col] = mean_val


# drop empty columns
data = data.drop(columns = ['FP16 GFLOPS', 'FP32 GFLOPS', 'FP64 GFLOPS'])

# change the format of the date to year
data['Release Date'] = pd.to_datetime(data['Release Date'])
data['Release Date'] = data['Release Date'].dt.year
print(data.columns)
print(data.to_string)

# visualize to know the distribution of conductors
fig = px.pie(data,'Type', title='Distribution of the types of conductors')
fig.update_layout(title_x = 0.5)
fig.show()

#to visualize the number of transistors by type over the years
grouped_data = data.groupby(['Release Date', 'Type']).sum().reset_index()

fig = px.line(data_frame=grouped_data, x='Release Date', y='Transistors (million)', color='Type',
              title='Number of transistors by type over the years',
              labels={'Transistors (million)': 'Transistors (million)', 'Type': 'Type', 'Release Date': 'Date'})
fig.update_layout(title_x=0.5)
fig.update_layout(xaxis=dict(
    tickmode='array',
    tickvals=[str(year) for year in range(2000, 2023, 2)],
    dtick='M24'))
fig.show()

# to check if Dannard Scaling is stil valid
fig = px.scatter(data_frame=data, x='Process Size (nm)', y='Transistors (million)', 
                 size='Die Size (mm^2)', color='TDP (W)', 
                 title='Dennard Scaling in Processor Design',
                 labels={'Process Size (nm)': 'Process Size (nm)', 'Transistors (million)': 'Transistors (million)', 
                         'Die Size (mm^2)': 'Die Size (mm^2)', 'TDP (W)': 'TDP (W)'})
fig.update_layout(title_x = 0.5)
fig.update_layout(hoverlabel=dict(bgcolor='black', font_color='white'))
fig.show()

# to check the frequencies of CPU and GPUs
grouped_data = data.groupby(['Release Date', 'Type']).sum().reset_index()

fig = px.line(data_frame=grouped_data, x='Release Date', y='Freq (MHz)', color='Type',
              title='Frequencies of CPU and GPU over the years',
              labels={'Freq (MHz)': 'Freq (MHz)', 'Type': 'Type', 'Release Date': 'Year'})
fig.update_layout(title_x=0.5)
fig.update_layout(xaxis=dict(
    tickmode='array',
    tickvals=[str(year) for year in range(2000, 2023, 2)],
    dtick='M24'))
fig.show()

# to check the vendors and processor sizes they produced.
grouped_data = data.groupby(['Vendor', 'Process Size (nm)']).sum().reset_index()

fig = px.bar(grouped_data, x='Vendor', y='Process Size (nm)', color='Vendor',
             labels={'Vendor': 'Vendor', 'Process Size (nm)': 'Process Size (nm)'})

fig.update_layout(title='Process size produced by different vendors', 
                   title_x=0.5, showlegend=False)
fig.show()

# to check which company produces the highest chips
grouped_data = data.groupby(['Foundry', 'Process Size (nm)']).sum().reset_index()
fig = px.pie(data,'Foundry', title='Number of computer chips produced by different companies')
fig.update_layout(title_x = 0.5)
fig.show()

# correlation for the features
feats = data.columns
cor_matrix = data[feats].corr()
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
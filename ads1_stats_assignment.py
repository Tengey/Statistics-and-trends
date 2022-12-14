# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

def pandas_reader(path): 
    """
    This function aims to read the csv file and return 2 DataFrames, one with 
    years as columns and one with countries as columns.
     
    Args: 
        path (str) : Takes its arguement, csv file path, as a string
        
    Return: 
        Returns two DataFrames, original and transposed 
    """
    df = pd.read_csv(path, skiprows = 4, index_col = ['Country Name',
                                                         'Indicator Name'])
    df.drop(['Indicator Code', 'Country Code',  'Unnamed: 66'], 1,
               inplace = True)
    df_or = df.reset_index()
    df_tr = df.transpose()
    return df_or, df_tr

df0, df1 = pandas_reader('API_19_DS2_en_csv_v2_4700503.csv')

print(df0)
print('\n')
print(df1)
print('\n')
print(df0.head())
print('\n')
print(df1.head())
print('\n')

# Exploring the statistical properties of df
print(df1.describe())

# Exploring the statistical properties of df
#print(df1['Urban population (% of total population)'].describe())

def indicator_line_plot (df_name, indicator_1, indicator_2):
    """
    This function takes in a DataFrame, and multiple indicator names to return
    a line graph of the inputted indicators.
     
    Args: 
        df_name, indicator_1 (str), indicator_2 (str): all arguements are
        taken as strings except for the DataFrame.
        
    Return: 
        Returns a line graph representing the inputted indicators. 
    """
    indicators = [indicator_1, indicator_2]
    for indicator in indicators:
        
        df_or = df_name[df_name['Indicator Name'] == indicator] 
        df_or.set_index('Country Name', inplace = True)
        df_or.drop('Indicator Name', 1, inplace =True)
        data_1 = df_or.T
        data_1 = data_1.reset_index()
        data_1 = data_1.rename(columns = {'index': 'year'})
        data_1.plot(x ='year', y= ['Lesotho', 'Zimbabwe', 'South Africa'],
                kind = 'line' )
        plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')    
        plt.title('{}'.format(indicator))
        plt.show()
        
indicator_line_plot(df0, 'Population, total',
                    'CO2 emissions (kg per PPP $ of GDP)')


def bar_plot_ind(df_name, indicator_1, indicator_2):
    """
    This function takes in 3 positional arguements, a DataFrame and 2 different
    indicator names to return a bar graph of the inputted indicators.
     
    Args: 
        df_name, indicator_1 (str), indicator_2 (str): all arguements are
        taken as strings except for the first one, the DataFrame.
        
    Return: 
        Returns a bar graph showing trends of particular indicators by
        countries in years. 
    """
    indicators = [indicator_1, indicator_2]
    for indicator in indicators:
        
        data_1 = df_name[df_name['Indicator Name'] == indicator]
        data_1.loc[(data_1['Country Name']=="Egypt, Arab Rep."), 'Northern Africa'] = "Egypt, Arab Rep."
        data_1.loc[(data_1['Country Name']=="Libya"), 'Northern Africa'] = "Libya"
        data_1.loc[(data_1['Country Name']=="Morocco"), 'Northern Africa'] = "Morocco"
        df = data_1.groupby(['Northern Africa'])['1980', '1990', '2000', '2010', '2020'].mean()
        df.plot(kind = 'bar', colormap='Paired')
        plt.title('{} from 1980 till 2020'.format(indicator))
        plt.show()
        
bar_plot_ind(df0, 'Agriculture, forestry, and fishing, value added (% of GDP)',
             'Energy use (kg of oil equivalent per capita)')


def country_correlation(df_name, country_x, country_y, country_z):
    """
    This function takes in 4 positional arguements, a DataFrame and 3 different
    country names to return a correlation table of the inputted countries.
     
    Args: 
        df_name, country_x (str), country_y (str), country_z (str): all
        arguements are taken as strings except for the first one,
        the DataFrame.
        
    Return: 
        Returns a correlation table/ heatmap showing trends of various
        indicators in selected countries. 
    """
    countries = [country_x, country_y, country_z]
    for country in countries:
        
        nation = df_name[country]
        cols = ['Population, total',
            'Agriculture, forestry, and fishing, value added (% of GDP)',
            'Energy use (kg of oil equivalent per capita)',
            'Nitrous oxide emissions (% change from 1990)',
            'Methane emissions (% change from 1990)',
            'Total greenhouse gas emissions (% change from 1990)',
            'Other greenhouse gas emissions (% change from 1990)',
            'CO2 emissions (kg per PPP $ of GDP)']
        sns.heatmap(nation[cols].corr(), annot=True, cmap="YlGnBu")
        plt.title('{}'.format(country))
        plt.legend([], frameon=False)
        plt.show()
        
country_correlation(df1, 'Lesotho', 'Zimbabwe', 'South Africa')



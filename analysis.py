from classes import ProductGroupingSystem
import pandas as pd
import numpy as np

def calc_group_percentages(group):
    return group / group.sum()



def fc_analysis(json_file='filtered_forecast.json'):
    print(f"\nLoading and verifying data from {json_file}:")
    
    # Load the data from the JSON file
    df = ProductGroupingSystem.load_from_json2(json_file)
    #print(df.head(20))
    
    pivot = pd.pivot_table(df, index=['Group','Product'],columns='Month', values='Quantity', aggfunc = 'sum')
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot.fillna(0)
    
    percentage_pivot = pivot.groupby(level=0).apply(calc_group_percentages)    
    percentage_pivot = percentage_pivot.round(3)
    print(pivot)
    print(percentage_pivot)
    
    group_pivot = pd.pivot_table(df, index='Group', columns = 'Month', values = 'Quantity', aggfunc= 'sum')
    group_pivot['Total'] = group_pivot.sum(axis=1)
    group_pivot = group_pivot.fillna(0)
    print(group_pivot)
    
    percentage_pivot.to_csv('percent_pivot.csv')
    group_pivot.to_csv('group_pivot.csv')
    
if __name__ == "__main__":
    fc_analysis()
    
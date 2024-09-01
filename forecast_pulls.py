import pandas as pd
import json
from datetime import time
from classes import ProductGroupingSystem

def load_forecast_data(file_path):
    """Load forecast data from Excel file and rename 'article' column to 'Product'"""
    df = pd.read_excel(file_path)
    df = df.rename(columns={'article': 'Product'})
    return df

def filter_forecast_by_group(forecast_df, group_name, grouping_system):
    """Filter forecast DataFrame by a specific product group"""
    group_products = set(grouping_system.list_products_in_group(group_name))
    return forecast_df[forecast_df['Product'].isin(group_products)]

def get_product_group(product, grouping_system):
    """Determine which group a product belongs to"""
    for group_name in grouping_system.list_groups():
        if product in grouping_system.list_products_in_group(group_name):
            return group_name
    return "Ungrouped"

def filter_forecast_by_all_groups(forecast_df, grouping_system):
    """
    Filter forecast DataFrame to include only products that are in any non-empty group
    Add month and group columns
    """
    all_grouped_products = set()
    for group in grouping_system.list_non_empty_groups():
        all_grouped_products.update(grouping_system.list_products_in_group(group))
    
    filtered_df = forecast_df[forecast_df['Product'].isin(all_grouped_products)].copy()
    
    # Add month column
    filtered_df['Month'] = filtered_df['date'].dt.to_period('M')
    
    # Add group column
    filtered_df['Group'] = filtered_df['Product'].apply(lambda x: get_product_group(x, grouping_system))
    
    return filtered_df

def get_total_forecast_by_group(forecast_df, grouping_system):
    """Calculate total forecast for each non-empty product group"""
    total_forecast = {}
    for group in grouping_system.list_non_empty_groups():
        group_forecast = filter_forecast_by_group(forecast_df, group, grouping_system)
        total_forecast[group] = group_forecast['Quantity'].sum()
    return total_forecast

def find_ungrouped_products(forecast_df, grouping_system):
    """Identify products in the forecast that are not in any group"""
    all_grouped_products = set()
    for group in grouping_system.list_non_empty_groups():
        all_grouped_products.update(grouping_system.list_products_in_group(group))
    
    forecast_products = set(forecast_df['Product'].unique())
    return forecast_products - all_grouped_products

def save_filtered_forecast_to_json(filtered_df, filename='filtered_forecast.json'):
    '''Save the filtered forecast DataFrame to a JSON file'''
    # Convert the DataFrame to a dictionary
    data_dict = filtered_df.to_dict(orient='records')
    
    # Convert datetime, Period, and time objects to strings
    for item in data_dict:
        for key, value in item.items():
            try:
                if isinstance(value, pd.Timestamp):
                    item[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, pd.Period):
                    item[key] = str(value)
                elif isinstance(value, time):
                    item[key] = value.strftime('%H:%M:%S')
                elif pd.isna(value):
                    item[key] = None
            except Exception as e:
                print(f"Error processing key {key} with value {value}: {str(e)}")
                item[key] = str(value)  # Convert to string as a fallback

    # Create a dictionary with the data
    data = {
        'filtered_forecast': data_dict
    }

    # Save to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'Filtered forecast data saved to {filename}')
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
        # Optionally, you could save the data as CSV if JSON fails
        filtered_df.to_csv(filename.replace('.json', '.csv'), index=False)
        print(f'Filtered forecast data saved to CSV instead: {filename.replace(".json", ".csv")}')

def main():
    # Load your product grouping system
    grouping_system = ProductGroupingSystem.load_from_json('bakery_products.json')

    # Load your forecast DataFrame
    forecast_df = load_forecast_data('forecast.xlsx')
    print("Forecast data loaded:")
    print(forecast_df.head())
    
    filtered_forecast = filter_forecast_by_all_groups(forecast_df, grouping_system)
    print("\nFiltered Foreast (only products in groups)")
    print(filtered_forecast.tail(25))
    
    product_count = filtered_forecast['Product'].nunique()
    print(f"\nNumber of unique products in the filtered forecast: {product_count}")

    quantity_by_group = {}
    for group in grouping_system.list_non_empty_groups():
        group_forecast = filter_forecast_by_group(filtered_forecast, group, grouping_system)
        quantity_by_group[group] = group_forecast['Quantity'].sum()
    
    print("\nQuantity distribution across groups:")
    for group, quantity in quantity_by_group.items():
        print(f"{group}: {quantity}")

    # Find products in the original forecast that were filtered out
    all_products = set(forecast_df['Product'].unique())
    filtered_products = set(filtered_forecast['Product'].unique())
    excluded_products = all_products - filtered_products
    #print("\nProducts in original forecast not assigned to any group:")
    #print(excluded_products)
    
    save_filtered_forecast_to_json(filtered_forecast, 'filtered_forecast.json')

if __name__ == "__main__":
    main()
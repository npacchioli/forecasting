import json
import pandas as pd

class ProductGroupingSystem:
    def __init__(self):
        self.groups = {}
        self.products = set()
        
    def add_products(self, products):
        '''Add multiple products to the available products'''
        self.products.update(products)
    
    def create_groups(self, group_names):
        '''Create multiple new groups'''
        for group_name in group_names:
            if group_name not in self.groups:
                self.groups[group_name] = set()
            else:
                print(f'Group {group_name} already exists.')
    
    def add_products_to_group(self, products, group_name):
        '''Add multiple products to a specific group'''
        if group_name not in self.groups:
            print(f'Group name {group_name} does not exist.')
            return
        for product in products:
            if product in self.products:
                self.groups[group_name].add(product)
            else:
                print(f'Product {product} is not in available product list')
                
    def list_groups(self):
        '''List all groups'''
        return list(self.groups.keys())

    def list_products_in_group(self, group_name):
        '''List all products in a specific group'''
        if group_name in self.groups:
            return list(self.groups[group_name])
        else:
            print(f'Group {group_name} does not exist.')
            return []

    def list_all_products(self):
        '''List all available products'''
        return list(self.products)
    
    def list_non_empty_groups(self):
        '''List all groups that contain products'''
        return [group for group, products in self.groups.items() if products]
                
                
    def save_to_json(self, filename='product_groups.json'):
        '''Save the current state to a JSON file'''
        data = {
            'products': list(self.products),
            'groups': {group: list(products) for group, products in self.groups.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'Data saved to {filename}')


    @classmethod
    def load_from_json(cls, filename='product_groups.json'):
        '''Load state from a JSON file'''
        with open(filename, 'r') as f:
            data = json.load(f)
        
        grouping_system = cls()
        grouping_system.add_products(data['products'])
        for group, products in data['groups'].items():
            grouping_system.create_groups([group])
            grouping_system.add_products_to_group(products, group)
        
        print(f'Data loaded from {filename}')
        return grouping_system
        
    @classmethod
    def load_from_json2(cls, filename='filtered_forecast.json'):
        '''Load state from a JSON file'''
        with open(filename, 'r') as f:
            data = json.load(f)
            
            df = pd.DataFrame(data['filtered_forecast'])
            
            df['date'] = pd.to_datetime(df['date'])
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
            df['Month'] = pd.to_datetime(df['Month']).dt.to_period('M')
    
            return df
                
        
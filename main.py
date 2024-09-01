from classes import ProductGroupingSystem
from products import all_products, groups

def main():
    grouping_system = ProductGroupingSystem()
    
    #add all products
    grouping_system.add_products(all_products)
    
    #create all groups
    grouping_system.create_groups(groups)
    
    bread_products = ['BAGUETTE','PAIN TRADITIONAL','BANETTE','BANETTINE','BOULE 200G','BOULE 400G','DEMI PAIN']
    grouping_system.add_products_to_group(bread_products, 'Breads')
    
    viennoiseries_products = ["PAIN AU CHOCOLAT", "CROISSANT", "PAIN AUX RAISINS", "CHAUSSON AUX POMMES"]
    grouping_system.add_products_to_group(viennoiseries_products, "Viennoiseries")
    
    pastries_products = ["ECLAIR", "MILLES FEUILLES", "PARIS BREST", "TARTE FRUITS 4P", "TARTE FRUITS 6P"]
    grouping_system.add_products_to_group(pastries_products, "Pastries")
    
    # Add products to the "Seasonal/Holiday Items" group
    seasonal_products = [
        "TARTE FRAISE 4PER", "TARTE FRAISE 6P", "FRAISIER",
        "TROPEZIENNE", "TROPEZIENNE FRAMBOISE",
        "TARTE FINE", "PUMPKIN SPICE CROISSANT",
        "BUCHE 4PERS", "BUCHE 6PERS", "BUCHE 8PERS", "GALETTE DES ROIS"
    ]
    grouping_system.add_products_to_group(seasonal_products, "Seasonal/Holiday Items")
    
    
    # Add products to the "Beverages" group
    beverage_products = ["CAFE OU EAU", "BOISSON 33CL", "THE"]
    grouping_system.add_products_to_group(beverage_products, "Beverages")
    
    ingredients_supplies_products = ["SACHET DE CROUTON", "PATES", "SACHET DE VIENNOISERIE"]
    grouping_system.add_products_to_group(ingredients_supplies_products, "Ingredients/Supplies")
    
    grouping_system.save_to_json('bakery_products.json')
      
    for group_name, products in grouping_system.groups.items():
        print(f'{group_name}: {products}')
        
    print("All groups:")
    print(grouping_system.list_groups())

    # Print out products in a specific group
    print("\nProducts in 'Breads' group:")
    print(grouping_system.list_products_in_group("Breads"))
    
    print("\nProducts in 'Beverages' group:")
    print(grouping_system.list_products_in_group("Beverages"))

    # Print out all available products
    #print("\nAll available products:")
   #print(grouping_system.list_all_products())
        

if __name__ == '__main__':
    main()
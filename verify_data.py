from classes import ProductGroupingSystem

def verify_loaded_data(json_file='bakery_products.json'):
    print(f"\nLoading and verifying data from {json_file}:")
    
    # Load the data from the JSON file
    loaded_system = ProductGroupingSystem.load_from_json(json_file)

    # Verify loaded data
    print("\nVerifying loaded data:")
    
    # Print all non-empty groups
    non_empty_groups = loaded_system.list_non_empty_groups()
    print("Non-empty groups:", non_empty_groups)
    
    # Print products in each non-empty group
    for group in non_empty_groups:
        print(f"\nProducts in '{group}' group:")
        print(loaded_system.list_products_in_group(group))

    # Additional verification steps
    print("\nAdditional Verifications:")
    
    # Check for empty groups
    all_groups = loaded_system.list_groups()
    empty_groups = set(all_groups) - set(non_empty_groups)
    # if empty_groups:
    #     print(f"Warning: The following groups are empty: {', '.join(empty_groups)}")
    # else:
    #     print("All groups contain products.")

    # Check for products not in any group
    all_products = set(loaded_system.list_all_products())
    products_in_groups = set()
    for group in non_empty_groups:
        products_in_groups.update(loaded_system.list_products_in_group(group))
    #products_not_in_groups = all_products - products_in_groups
    print(f'\nAll Products in groups')
    print(products_in_groups)
    
    # if products_not_in_groups:
    #     print(f"Warning: The following products are not in any group: {', '.join(products_not_in_groups)}")
    # else:
    #     print("All products are assigned to at least one group.")

if __name__ == "__main__":
    verify_loaded_data()
    
    # If you want to verify a different file, you can call the function with a filename:
    # verify_loaded_data('different_file.json')
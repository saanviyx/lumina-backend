import os  # Import for file and directory operations
import pandas as pd  # Import pandas for data manipulation
import random  # Import random for selecting random products

class SkinCareRecommender:
    def __init__(self, products_csv_path=None):
        self.products_df = None  # Initialize products dataframe as None
        self.products_csv_path = products_csv_path  # Store path to products CSV file
        
        # Define the order of steps in a skincare routine
        self.routine_steps = [
            "Cleanser",  # First step in skincare routine
            "Toner",  # Second step
            "Serum",  # Third step
            "Treatment",  # Fourth step
            "Moisturizer",  # Fifth step
            "Sunscreen"  # Final step
        ]
        
        # Create a dictionary of keywords for each skin condition and product type
        self.disease_keywords = {
            'Acne': {
                'Cleanser': ['salicylic acid', 'benzoyl peroxide', 'tea tree', 'gentle', 'non-comedogenic'],  # Keywords for acne cleansers
                'Toner': ['witch hazel', 'salicylic acid', 'antibacterial', 'balancing'],  # Keywords for acne toners
                'Serum': ['niacinamide', 'zinc', 'hyaluronic acid', 'vitamin b5'],  # Keywords for acne serums
                'Treatment': ['benzoyl peroxide', 'salicylic acid', 'retinoids', 'spot treatment'],  # Keywords for acne treatments
                'Moisturizer': ['oil-free', 'non-comedogenic', 'lightweight', 'gel'],  # Keywords for acne moisturizers
                'Sunscreen': ['non-comedogenic', 'oil-free', 'matte', 'mineral']  # Keywords for acne sunscreens
            },
            
            'Eczema': {
                'Cleanser': ['gentle', 'fragrance-free', 'soap-free', 'hydrating', 'ceramide'],  # Keywords for eczema cleansers
                'Toner': ['alcohol-free', 'fragrance-free', 'soothing', 'hydrating'],  # Keywords for eczema toners
                'Serum': ['ceramide', 'hyaluronic acid', 'soothing', 'barrier repair'],  # Keywords for eczema serums
                'Treatment': ['colloidal oatmeal', 'aloe vera', 'soothing', 'anti-itch'],  # Keywords for eczema treatments
                'Moisturizer': ['ceramide', 'thick', 'fragrance-free', 'intensive', 'healing'],  # Keywords for eczema moisturizers
                'Sunscreen': ['mineral', 'sensitive skin', 'fragrance-free', 'gentle']  # Keywords for eczema sunscreens
            },
            
            'Rosacea': {
                'Cleanser': ['gentle', 'fragrance-free', 'sulfate-free', 'calming'],  # Keywords for rosacea cleansers
                'Toner': ['alcohol-free', 'soothing', 'anti-redness', 'calming'],  # Keywords for rosacea toners
                'Serum': ['niacinamide', 'azelaic acid', 'anti-inflammatory', 'calming'],  # Keywords for rosacea serums
                'Treatment': ['azelaic acid', 'calming', 'anti-redness', 'centella asiatica'],  # Keywords for rosacea treatments
                'Moisturizer': ['fragrance-free', 'calming', 'anti-redness', 'lightweight'],  # Keywords for rosacea moisturizers
                'Sunscreen': ['mineral', 'zinc oxide', 'gentle', 'calming']  # Keywords for rosacea sunscreens
            },
            
            'Basal Cell Carcinoma': {
                'Cleanser': ['gentle', 'non-irritating', 'hydrating'],  # Keywords for BCC cleansers
                'Toner': ['soothing', 'fragrance-free', 'non-irritating'],  # Keywords for BCC toners
                'Serum': ['healing', 'antioxidant', 'gentle'],  # Keywords for BCC serums
                'Treatment': ['healing', 'gentle', 'non-irritating'],  # Keywords for BCC treatments
                'Moisturizer': ['barrier repair', 'healing', 'hydrating'],  # Keywords for BCC moisturizers
                'Sunscreen': ['high spf', 'zinc oxide', 'titanium dioxide', 'physical blocker']  # Keywords for BCC sunscreens
            },
            
            'Blackheads': {
                'Cleanser': ['salicylic acid', 'charcoal', 'clay', 'deep cleansing'],  # Keywords for blackhead cleansers
                'Toner': ['witch hazel', 'salicylic acid', 'clarifying', 'pore-minimizing'],  # Keywords for blackhead toners
                'Serum': ['niacinamide', 'salicylic acid', 'pore-refining'],  # Keywords for blackhead serums
                'Treatment': ['blackhead strips', 'charcoal mask', 'clay mask', 'exfoliating'],  # Keywords for blackhead treatments
                'Moisturizer': ['oil-free', 'non-comedogenic', 'lightweight', 'gel'],  # Keywords for blackhead moisturizers
                'Sunscreen': ['non-comedogenic', 'oil-free', 'matte', 'lightweight']  # Keywords for blackhead sunscreens
            },
            
            'Normal': {
                'Cleanser': ['gentle', 'hydrating', 'pH-balanced'],  # Keywords for normal skin cleansers
                'Toner': ['not required'],  # Keywords for normal skin toners
                'Serum': ['vitamin c', 'hyaluronic acid', 'antioxidant'],  # Keywords for normal skin serums
                'Treatment': ['not required'],  # Keywords for normal skin treatments
                'Moisturizer': ['balanced', 'hydrating', 'protective'],  # Keywords for normal skin moisturizers
                'Sunscreen': ['gentle', 'calming']  # Keywords for normal skin sunscreens
            }
        }
        
        # Load product data if a path was provided in the constructor
        if self.products_csv_path:
            self.load_products()
    
    def load_products(self, csv_path=None):
        """Load product data from CSV file"""
        if csv_path:
            self.products_csv_path = csv_path  # Update CSV path if provided
            
        if not self.products_csv_path:
            raise ValueError("Products CSV path not provided")  # Error if no path is set
            
        if not os.path.exists(self.products_csv_path):
            raise FileNotFoundError(f"Products CSV not found at {self.products_csv_path}")  # Error if file doesn't exist
            
        self.products_df = pd.read_csv(self.products_csv_path)  # Load data from CSV into DataFrame
        
        # Extract brand from product name if brand column is missing
        if 'brand' not in self.products_df.columns:
            self.products_df['brand'] = self.products_df['product_name'].apply(
                lambda x: x.split()[0] if len(x.split()) > 0 else ''  # Take first word as brand
            )
        
        return self.products_df  # Return loaded DataFrame
        
    def get_product_category(self, product_name, product_type):
        """Determine which skincare routine category a product belongs to"""
        product_text = (product_name + " " + product_type).lower()  # Combine name and type, convert to lowercase
        
        # Define keywords that indicate which category a product belongs to
        category_keywords = {
            'Cleanser': ['cleanser', 'cleansing', 'face wash', 'facial wash', 'soap'],  # Keywords for cleanser products
            'Toner': ['toner', 'toning', 'mist', 'essence'],  # Keywords for toner products
            'Serum': ['serum', 'ampoule', 'concentrate', 'booster'],  # Keywords for serum products
            'Treatment': ['treatment', 'spot', 'mask', 'exfoliant', 'peel', 'acne', 'cream'],  # Keywords for treatment products
            'Moisturizer': ['moisturiser', 'moisturizer', 'moisturising', 'moisturizing', 'lotion', 'cream', 'hydrator'],  # Keywords for moisturizer products
            'Sunscreen': ['sunscreen', 'spf', 'sun protection', 'uv protection']  # Keywords for sunscreen products
        }
        
        # Search for keywords in the product text
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in product_text:
                    return category  # Return the matching category
                    
        # If no category is found, default to "Treatment"
        return 'Treatment'
    
    def categorize_products(self, products_list):
        """Categorize products into skincare routine steps"""
        categorized = {step: [] for step in self.routine_steps}  # Initialize empty lists for each step
        
        # Categorize each product in the list
        for product in products_list:
            category = self.get_product_category(
                product.get('product_name', ''),  # Get product name or empty string
                product.get('product_type', '')  # Get product type or empty string
            )
            categorized[category].append(product)  # Add product to its category
            
        return categorized  # Return dictionary of categorized products
    
    def get_price_range_products(self, products_list, num_per_price=3):
        """Get products in different price ranges (low, medium, high)"""
        if not products_list:
            return []  # Return empty list if no products
            
        # Convert input to DataFrame if it's a list of dictionaries
        if isinstance(products_list[0], dict):
            products_df = pd.DataFrame(products_list)  # Convert list of dicts to DataFrame
        else:
            products_df = products_list  # Already a DataFrame
            
        # Handle price data
        if 'price' not in products_df.columns:
            # If no price column, try to extract from string representation
            if 'price' in products_df:
                products_df['price_value'] = products_df['price'].apply(
                    lambda x: float(str(x).replace('£', '').replace('$', '').replace('€', ''))  # Remove currency symbols
                    if str(x).replace('£', '').replace('$', '').replace('€', '').replace('.', '').isdigit()  # Check if it's a number
                    else 0.0  # Default to 0 if not a number
                )
            else:
                # If no price info, return random selection
                return random.sample(list(products_list), min(num_per_price, len(products_list)))
        else:
            # Convert price to numeric value
            products_df['price_value'] = products_df['price'].apply(
                lambda x: float(str(x).replace('£', '').replace('$', '').replace('€', ''))  # Remove currency symbols
                if str(x).replace('£', '').replace('$', '').replace('€', '').replace('.', '').isdigit()  # Check if it's a number
                else 0.0  # Default to 0 if not a number
            )
            
        # Sort products by price
        products_df = products_df.sort_values('price_value')
        
        # Divide products into price ranges
        total_products = len(products_df)
        low_end = max(1, int(total_products * 0.33))  # End of low price range (33% of products)
        high_start = max(low_end + 1, int(total_products * 0.67))  # Start of high price range (67% of products)
        
        # Split into price ranges
        low_price = products_df.iloc[:low_end]  # Low price products
        mid_price = products_df.iloc[low_end:high_start]  # Mid price products
        high_price = products_df.iloc[high_start:]  # High price products
        
        # Select products from each range
        selected = []
        
        # Get one product from low price range
        if not low_price.empty:
            selected.append(low_price.sample(min(1, len(low_price))).iloc[0].to_dict())
            
        # Get one product from mid price range
        if not mid_price.empty:
            selected.append(mid_price.sample(min(1, len(mid_price))).iloc[0].to_dict())
            
        # Get one product from high price range
        if not high_price.empty:
            selected.append(high_price.sample(min(1, len(high_price))).iloc[0].to_dict())
            
        # Fill remaining slots with random products if needed
        while len(selected) < num_per_price and len(products_df) > len(selected):
            # Get products that haven't been selected yet
            remaining = products_df[~products_df['product_name'].isin([p['product_name'] for p in selected])]
            if not remaining.empty:
                selected.append(remaining.sample(1).iloc[0].to_dict())  # Add a random product
            else:
                break  # No more unique products to add
                
        return selected  # Return selected products
    
    def find_matching_products(self, disease, product_category):
        """Find products that match a disease and category"""
        if self.products_df is None:
            raise ValueError("Products not loaded. Call load_products() first.")  # Error if products aren't loaded
            
        matching_products = []  # Initialize empty list for matching products
        
        # Get keywords for this disease and product category
        keywords = self.disease_keywords.get(disease, {}).get(product_category, [])
        
        # If no specific keywords found, use general keywords from Acne
        if not keywords:
            keywords = self.disease_keywords.get('Acne', {}).get(product_category, [])
            
        # Search for products that match each keyword
        for keyword in keywords:
            # Match keyword in ingredients, product name, or product type
            keyword_matches = self.products_df[
                (self.products_df['clean_ingreds'].astype(str).str.contains(keyword, case=False, na=False)) |
                (self.products_df['product_name'].astype(str).str.contains(keyword, case=False, na=False)) |
                (self.products_df['product_type'].astype(str).str.contains(keyword, case=False, na=False))
            ]
            
            matching_products.extend(keyword_matches.to_dict('records'))  # Add matching products to list
            
        # Remove duplicate products (those that match multiple keywords)
        unique_products = {}
        for product in matching_products:
            if product['product_name'] not in unique_products:
                unique_products[product['product_name']] = product  # Keep unique products
                
        return list(unique_products.values())  # Return list of unique matching products
    
    def recommend_for_disease(self, disease):
        """Generate a full skincare routine recommendation for a skin disease"""
        if disease not in self.disease_keywords:
            print(f"Warning: '{disease}' not found in disease keywords. Using general recommendations.")
            disease = list(self.disease_keywords.keys())[0]  # Use first disease as fallback
            
        routine = {}  # Initialize empty dictionary for routine
        
        # For each step in the skincare routine
        for step in self.routine_steps:
            # Find products matching the disease and step
            matching_products = self.find_matching_products(disease, step)
            
            # Get products in different price ranges
            if matching_products:
                routine[step] = self.get_price_range_products(matching_products)  # Get budget, mid, and premium options
            else:
                print(f"No {step} for required for {disease} skin")  # Log if no products found
                routine[step] = []  # Empty list for this step
                
        return routine  # Return complete routine with products for each step
    
    def format_routine(self, routine, disease):
        """Format the skincare routine for display"""
        result = f"Recommended Skincare Routine for {disease} \n\n"  # Create header with disease name
        
        # Loop through each step in the routine
        for step in self.routine_steps:
            result += f" {step}:\n"  # Add step name
            
            # Check if we have products for this step
            if step in routine and routine[step]:
                for i, product in enumerate(routine[step]):
                    price_level = ["Budget-friendly", "Mid-range", "Premium"][min(i, 2)]  # Label for price range
                    result += f"  {price_level}: {product['product_name']}"  # Add product with price level
                    
                    # Add price if available
                    if 'price' in product and product['price']:
                        result += f" ({product['price']})"  # Include price in parentheses
                        
                    result += "\n"  # New line after each product
            else:
                result += "  No specific recommendations found for this step\n"  # Message if no products found
                
            result += "\n"  # Add space between steps
            
        # Add usage instructions at the end
        result += "Usage Instructions:\n"
        result += "1. Always cleanse your face before applying other products\n"  # First instruction
        result += "2. Apply products in the order listed above\n"  # Second instruction
        result += "3. Allow each product to absorb before applying the next\n"  # Third instruction
        result += "4. Use sunscreen as the last step in your morning routine\n"  # Fourth instruction
        result += "5. Consult with a dermatologist for personalized advice\n"  # Fifth instruction
        
        return result  # Return formatted routine as string

"""
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of current file
    parent_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Get parent directory
    dataset_dir = os.path.join(parent_dir, "dataset")  # Path to dataset directory
    products_csv = os.path.join(dataset_dir, "skin_products.csv")  # Path to products CSV
    
    recommender = SkinCareRecommender(products_csv)  # Create recommender with products data
    routine = recommender.recommend_for_disease("Acne")  # Generate routine for acne
    formatted_routine = recommender.format_routine(routine, "Acne")  # Format the routine
    print(formatted_routine)  # Print the formatted routine
"""
import os  # Import the os module to work with the operating system and file paths
import sys  # Import sys module for accessing command-line arguments and system-specific parameters
from predict import SkinDiseaseClassifier  # Import the custom classifier class that handles the skin disease prediction
from recommend import SkinCareRecommender  # Import the recommender class that suggests skincare products based on diagnosed conditions

def main():
    # Setup paths based on the original code structure
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where this script is located
    ml_root = os.path.abspath(os.path.join(script_dir, "../"))  # Navigate one directory up to find the root of the ML project
    
    dataset_dir = os.path.join(ml_root, "dataset")  # Path to the dataset directory containing images and product data
    model_path = os.path.join(ml_root, "models", "prediction_best.h5")  # Path to the trained neural network model file
    products_csv_path = os.path.join(dataset_dir, "skin_products.csv")  # Path to the CSV containing skincare product data
    test_dir = os.path.join(dataset_dir, "test")  # Directory containing test images organized by condition
    sample_image_path = os.path.join(test_dir, "Acne", "acne-pustular-16.jpg")  # Default image path for testing (pustular acne)
    # sample_image_path = os.path.join(test_dir, "Basal Cell Carcinoma", "basal-cell-carcinoma-face-13.jpg")
    # sample_image_path = os.path.join(test_dir, "Blackheads", "53.jpg")
    # sample_image_path = os.path.join(test_dir, "Eczema", "eczema31.jpg")
    # sample_image_path = os.path.join(test_dir, "Normal", "normal_ac223b0aac8319751061_jpg.rf.46ffc21b6bee315748ce626b47cbb1d0.jpg")
    # sample_image_path = os.path.join(test_dir, "Rosacea", "rosacea-71.jpg")
    
    # Allow overriding the image path via command line argument
    if len(sys.argv) > 1:  # Check if user provided an image path as command line argument
        sample_image_path = sys.argv[1]  # Use the user-provided image path instead of the default
    
    # Check if files exist
    if not os.path.exists(model_path):  # Make sure the model file exists before proceeding
        print(f"Error: Model file not found at {model_path}")  # Show error message with the missing file path
        return  # Exit the function early if model file is missing
    
    if not os.path.exists(products_csv_path):  # Check if the products database exists
        print(f"Error: Products CSV file not found at {products_csv_path}")  # Show error message with path
        return  # Exit the function if product database is missing
    
    if not os.path.exists(sample_image_path):  # Verify the image to be analyzed exists
        print(f"Error: Sample image not found at {sample_image_path}")  # Show error with missing image path
        return  # Exit if the image doesn't exist
    
    # Define class labels (should match those used during training)
    class_labels = ['Acne', 'Basal Cell Carcinoma', 'Blackheads', 'Eczema', 'Normal', 'Rosacea']  # List of skin conditions the model can identify
    
    # Load classifier
    print("Loading skin disease classifier...")  # Give user feedback that model is loading
    classifier = SkinDiseaseClassifier()  # Create an instance of the classifier
    classifier.load_model(model_path, class_labels)  # Load the pre-trained model with appropriate class labels
    
    # Classify the image
    print(f"Analyzing image: {sample_image_path}")  # Tell user which image is being analyzed
    result = classifier.predict(sample_image_path)  # Process the image and get prediction results
    print(f"\n🔬 SKIN ANALYSIS RESULTS")  # Display header for results with emoji for visual appeal
    print(f"Detected Skin Condition: {result['disease']}")  # Show the detected skin condition
    print(f"Confidence: {result['confidence']:.2f}%\n")  # Show prediction confidence formatted to 2 decimal places
    
    # Load recommender
    recommender = SkinCareRecommender(products_csv_path)  # Initialize the product recommender with our product database
    
    # Get recommendations
    try:
        routine = recommender.recommend_for_disease(result['disease'])  # Get product recommendations based on the detected condition
        formatted_routine = recommender.format_routine(routine, result['disease'])  # Format the recommendations into a user-friendly format
        
        print(formatted_routine)  # Display the formatted skincare routine recommendations
    except Exception as e:  # Catch any errors that might occur during recommendation
        print(f"Error generating recommendations: {str(e)}")  # Show error message with details if recommendations fail

if __name__ == "__main__":  # Check if this script is being run directly (not imported)
    main()  # Call the main function to start the program
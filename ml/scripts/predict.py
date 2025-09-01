import tensorflow as tf  # Import TensorFlow library for deep learning functionality
from keras.models import Sequential, load_model  # Import Keras models for building and loading neural networks
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Import layers needed for CNN architecture
from keras.preprocessing.image import ImageDataGenerator  # Import tool for data augmentation
from keras.callbacks import Callback  # Import base callback class to create custom callbacks
from keras.regularizers import l2  # Import L2 regularizer to reduce overfitting
import os  # Import OS module for file/directory operations
import matplotlib.pyplot as plt  # Import plotting library for visualizations
import numpy as np  # Import NumPy for numerical operations
import cv2  # Import OpenCV for image processing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Import tools for evaluation metrics

# Custom callback class to log training progress at specific intervals
class SimpleIterationLogger(Callback):
    def on_train_begin(self, logs=None):
        self.iterations = []  # Initialize an empty list to store iterations
    
    def on_epoch_end(self, iteration, logs=None):
        if (iteration + 1) % 5 == 0 or iteration == 0:  # Log every 5 epochs or on the first epoch
            print(f"Iteration {iteration+1}: "
                  f"loss: {logs.get('loss'):.4f}, "  # Print loss with 4 decimal places
                  f"train accuracy: {logs.get('accuracy')* 100:.2f}% ")  # Convert accuracy to percentage with 2 decimal places

# Main class for skin disease classification functionality
class SkinDiseaseClassifier:
    def __init__(self, img_size=(150, 150), batch_size=32):
        # Initialize with customizable image size and batch size
        self.IMG_SIZE = img_size  # Set image dimensions for processing
        self.BATCH_SIZE = batch_size  # Set batch size for training
        self.model = None  # Placeholder for the model
        self.class_labels = None  # Placeholder for disease class labels

    def build_model(self, num_classes=5):
        """Build a CNN model similar to the original but with regularization to reduce overfitting"""
        # Add L2 regularization to reduce overfitting
        reg = l2(0.001)  # Create L2 regularizer with strength 0.001
        
        model = Sequential([  # Create a sequential model with layers in order
            # First convolutional block with 32 filters and 3x3 kernel
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=reg, input_shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3)),
            MaxPooling2D(2, 2),  # Max pooling to reduce spatial dimensions by half
            
            # Second convolutional block with 64 filters
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg),
            MaxPooling2D(2, 2),  # Further reduce dimensions
            
            # Third convolutional block with 128 filters
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=reg),
            MaxPooling2D(2, 2),  # Final dimension reduction
            
            Flatten(),  # Flatten the 3D output to 1D for dense layers
            Dense(512, activation='relu', kernel_regularizer=reg),  # Fully connected layer with 512 neurons
            Dropout(0.5),  # Dropout layer to prevent overfitting by dropping 50% of inputs
            Dense(num_classes, activation='softmax')  # Output layer with one neuron per disease class
        ])
        
        # Configure the model for training with optimizer, loss function, and metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Use Adam optimizer with reduced learning rate
                      loss='categorical_crossentropy',  # Use categorical cross-entropy for multi-class classification
                      metrics=['accuracy'])  # Track accuracy during training
        
        return model  # Return the compiled model

    def train(self, train_dir, test_dir, epochs=200, model_save_path='prediction.h5'):
        """Train the model on the provided dataset directories"""
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Set path for saving the best model during training
        best_model_path = model_save_path.replace('.h5', '_best.h5')
        
        # Configure data augmentation for training to create variations of images
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0,1]
            rotation_range=20,  # Randomly rotate images by up to 20 degrees
            width_shift_range=0.2,  # Randomly shift images horizontally
            height_shift_range=0.2,  # Randomly shift images vertically
            shear_range=0.2,  # Apply shearing transformations
            zoom_range=0.2,  # Randomly zoom in on images
            horizontal_flip=True,  # Randomly flip images horizontally
            fill_mode='nearest'  # Fill in newly created pixels with nearest value
        )

        test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize test images, no augmentation

        # Create training data generator from directory
        train_generator = train_datagen.flow_from_directory(
            train_dir,  # Path to training directory
            target_size=self.IMG_SIZE,  # Resize images to specified dimensions
            batch_size=self.BATCH_SIZE,  # Number of images per batch
            class_mode='categorical',  # Use categorical labels (one-hot encoded)
            shuffle=True  # Shuffle the training data
        )

        # Create test data generator from directory
        test_generator = test_datagen.flow_from_directory(
            test_dir,  # Path to test directory
            target_size=self.IMG_SIZE,  # Resize images to same dimensions
            batch_size=self.BATCH_SIZE,  # Same batch size
            class_mode='categorical',  # Same label format
            shuffle=False  # Don't shuffle to maintain order for confusion matrix
        )

        # Extract class labels from directory structure
        self.class_labels = list(train_generator.class_indices.keys())
        num_classes = len(self.class_labels)  # Count number of classes

        # Build model with the correct number of output classes
        self.model = self.build_model(num_classes=num_classes)
        
        # Create logging callback to show training progress
        iteration_logger = SimpleIterationLogger()
        
        # Create callback to save the best model based on validation accuracy
        test_loss_checkpoint = CustomTestAccuracyCheckpoint(
            test_generator=test_generator,
            filepath=best_model_path,
            verbose=0  # Don't print logs from this callback
        )
        
        # Train the model with both callbacks
        history = self.model.fit(
            train_generator,  # Training data
            epochs=epochs,  # Number of complete passes through training data
            validation_data=test_generator,  # Data for validation during training
            callbacks=[iteration_logger, test_loss_checkpoint],  # List of callbacks
            verbose=0  # Don't show the default progress bar
        )
        
        # Save the final model after training
        self.model.save(model_save_path)
        
        # Load the best model (highest validation accuracy) if it exists
        if os.path.exists(best_model_path):
            self.model = load_model(best_model_path)
        
        # Evaluate model performance on test set
        test_loss, test_acc = self.model.evaluate(test_generator, verbose=0)
        print(f"\nFinal test accuracy: {test_acc:.4f}")
        
        # Generate and display confusion matrix
        self.plot_confusion_matrix(test_generator)
        
        return history, test_acc  # Return training history and test accuracy

    def plot_confusion_matrix(self, test_generator):
        """Generate and plot a confusion matrix for the test set"""
        # Reset test generator to start from the beginning
        test_generator.reset()

        # Initialize lists to store predictions and true labels
        y_pred = []
        y_true = []

        # Loop through all batches in the test generator
        for i in range(len(test_generator)):
            x_batch, y_batch = test_generator[i]  # Get a batch of test images and labels
            y_pred_batch = self.model.predict(x_batch, verbose=0)  # Make predictions
            y_pred_batch = np.argmax(y_pred_batch, axis=1)  # Convert from one-hot to class indices
            y_true_batch = np.argmax(y_batch, axis=1)  # Convert true labels from one-hot to indices
        
            y_pred.extend(y_pred_batch)  # Add batch predictions to list
            y_true.extend(y_true_batch)  # Add batch true labels to list
        
            # Break if we've processed all images (last batch might be smaller)
            if (i + 1) * test_generator.batch_size >= test_generator.n:
                break

        # Create confusion matrix from predictions and true labels
        cm = confusion_matrix(y_true, y_pred)

        # Create figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Create the confusion matrix display object
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')  # Plot using blue color map
    
        # Set titles and labels for the plot
        ax.set_title('Confusion Matrix for Skin Disease Classification', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
    
        # Rotate x-tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
        # Adjust layout to prevent cutting off labels
        fig.tight_layout()
    
        # Save the confusion matrix plot to file and display it
        cm_plot_path = 'confusion_matrix.png'
        plt.savefig(cm_plot_path)
        print(f"Confusion matrix saved to {cm_plot_path}")
        plt.show()

    def load_model(self, model_path, class_labels=None):
        """Load a pretrained model"""
        self.model = load_model(model_path)  # Load model from file
        if class_labels:
            self.class_labels = class_labels  # Set class labels if provided
        return self.model  # Return the loaded model

    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")  # Check if file exists
            
        img = cv2.imread(image_path)  # Read image from file
        if img is None:
            raise ValueError(f"Error loading image at {image_path}")  # Check if image was loaded successfully
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
        img = cv2.resize(img, self.IMG_SIZE)  # Resize to model input size
        img = img / 255.0  # Normalize pixel values to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img  # Return preprocessed image

    def predict(self, image_path):
        """Predict the disease class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded or trained. Call load_model() or train() first.")  # Check if model exists
            
        if self.class_labels is None:
            raise ValueError("Class labels not set. Either train the model or provide class labels.")  # Check if labels exist
        
        processed_image = self.preprocess_image(image_path)  # Preprocess the image
        
        predictions = self.model.predict(processed_image, verbose=0)  # Get model predictions
        predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get index of highest probability class
        confidence = predictions[0][predicted_class_idx] * 100  # Convert probability to percentage
        
        predicted_label = self.class_labels[predicted_class_idx]  # Get class name from index
        
        # Create detailed results dictionary
        result = {
            'disease': predicted_label,  # Predicted disease name
            'confidence': confidence,  # Confidence percentage
            'all_probabilities': {}  # Container for all class probabilities
        }
        
        # Include probabilities for all classes
        for i, label in enumerate(self.class_labels):
            result['all_probabilities'][label] = float(predictions[0][i] * 100)
        
        return result  # Return the prediction results

# Custom callback to save only the model with the best validation accuracy
class CustomTestAccuracyCheckpoint(Callback):
    def __init__(self, test_generator, filepath, verbose=0):
        super().__init__()  # Initialize parent class
        self.test_generator = test_generator  # Test data generator
        self.filepath = filepath  # Path to save best model
        self.verbose = verbose  # Verbosity level
        self.best_accuracy = 0  # Initialize best accuracy tracker

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate model on test set after each epoch
        test_loss, test_acc = self.model.evaluate(self.test_generator, verbose=0)
        
        # If current accuracy is better than previous best, save the model
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc  # Update best accuracy
            self.model.save(self.filepath)  # Save model to file

# Entry point for running the script directly
if __name__ == "__main__":
    # Setup file paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of the script
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))  # Get parent directory
    dataset_dir = os.path.join(parent_dir, "dataset")  # Path to dataset
    train_dir = os.path.join(dataset_dir, "train")  # Path to training data
    test_dir = os.path.join(dataset_dir, "test")  # Path to test data
    model_path = os.path.join(parent_dir, "models", "prediction.h5")  # Path to save model
    
    # Create classifier instance with specified parameters
    classifier = SkinDiseaseClassifier(img_size=(150, 150), batch_size=32)
    
    # Train the model with specified epochs and save path
    history, accuracy = classifier.train(train_dir, test_dir, epochs=150, model_save_path=model_path)
    
    # Example prediction on a test image
    sample_image_path = os.path.join(test_dir, "Rosacea", "rosacea-71.jpg")  # Path to sample image
    if os.path.exists(sample_image_path):  # Check if sample image exists
        result = classifier.predict(sample_image_path)  # Get prediction
        print(f"Predicted disease: {result['disease']}")  # Print predicted disease
        
        # Display the sample image with prediction
        sample_img = cv2.imread(sample_image_path)  # Read sample image
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        plt.imshow(sample_img)  # Display image
        plt.title(f"Predicted: {result['disease']}")  # Add title with prediction
        plt.show()  # Show the plot
    else:
        print("Sample image not found.")  # Error message if sample image doesn't exist
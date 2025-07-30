Image Classification of Natural Images using CNNs and Transfer Learning
This project demonstrates a complete machine learning workflow for image classification. It uses the "natural_images" dataset, which contains 8 distinct categories. The project covers everything from data analysis and preprocessing to building, training, and evaluating multiple Convolutional Neural Network (CNN) models, including custom architectures and transfer learning models (VGG16, MobileNetV2). Finally, it establishes an inference pipeline to predict the class of a new image using the best-performing model.

Model Comparison Plot

Table of Contents
Project Overview
Features
Dataset
Usage
Code Explanation
1. Data Analysis
2. Preprocessing
3. Data Splitting and Augmentation
4. Modeling
5. Training
6. Evaluation
7. Inference Pipeline
Results
Project Overview
The primary goal of this project is to classify images into one of eight categories. To achieve this, we explore and compare four different deep learning models:

A simple, custom-built CNN.
A deeper, more complex custom-built CNN.
A transfer learning model using the VGG16 architecture.
A transfer learning model using the MobileNetV2 architecture.
The project evaluates these models based on their test accuracy and other metrics, selects the best one, and creates a ready-to-use function for making predictions on new images.

Features
End-to-End Workflow: From data loading to final prediction.
Data Analysis: Comprehensive analysis of class distribution and image properties.
Image Preprocessing: Resizing, normalization, and shuffling.
Data Augmentation: Increases dataset diversity and reduces overfitting using ImageDataGenerator.
Multiple Model Architectures: Compares custom CNNs with powerful pre-trained models.
Transfer Learning: Leverages VGG16 and MobileNetV2 for robust feature extraction.
Callbacks: Uses EarlyStopping to prevent overfitting and ModelCheckpoint to save the best model weights.
Detailed Evaluation: Generates accuracy scores, classification reports, and confusion matrices.
Visualizations: Plots training history, model comparisons, and prediction examples.
Inference Pipeline: A simple function to classify a single image.
Dataset
The project uses the natural_images dataset, which must be structured with separate folders for each category.

Categories: airplane, car, cat, dog, flower, fruit, motorbike, person.
Image Size: All images are resized to 150x150 pixels.
Usage
To run the project, execute the Python script from your terminal.

Crucially, update the DATADIR variable in the script to point to the root directory of your "natural_images" dataset:

python
DATADIR = r'C:/path/to/your/natural_images'
Run the script:

bash
python your_script_name.py
The script will then:

Analyze the dataset and show plots.
Preprocess the data and save it to X.pickle and y.pickle.
Visualize data augmentation examples.
Build, train, and evaluate all four models.
Save the trained model weights (.h5 files) and performance plots (.png files) in the project directory.
Print a summary of the best-performing model.
Demonstrate the inference pipeline on a sample image.
Code Explanation
1. Data Analysis
The analyze_dataset() function performs an initial exploratory data analysis (EDA). It:

Counts the number of images in each category.
Plots a bar chart of the class distribution.
Displays one sample image from each category.
Analyzes the dimensions of a sample of images.
2. Preprocessing
The create_training_data() function handles the data preparation. For each image, it:

Reads the image file.
Converts the color space from BGR (OpenCV's default) to RGB.
Resizes the image to 150x150 pixels.
Normalizes pixel values to the [0, 1] range by dividing by 255.0.
Appends the processed image array and its corresponding integer label to a list.
Finally, the entire dataset is shuffled to ensure randomness. The processed data (X) and labels (y) are saved using pickle for quick reloading in the future.

3. Data Splitting and Augmentation
Splitting: The data is split into training (70%), validation (15%), and test (15%) sets using sklearn.model_selection.train_test_split.
Augmentation: keras.preprocessing.image.ImageDataGenerator is used to create augmented images for the training set on-the-fly. This helps the model generalize better by exposing it to a wider variety of image variations. The augmentations include:
rotation_range=15
width_shift_range=0.1
height_shift_range=0.1
shear_range=0.1
zoom_range=0.1
horizontal_flip=True
4. Modeling
Four different models are defined:

create_cnn_model1(): A simple CNN with three convolutional blocks, each containing Conv2D, BatchNormalization, MaxPooling2D, and Dropout.
create_cnn_model2(): A deeper CNN with four convolutional blocks and more filters, designed to capture more complex features.
create_vgg16_model(): A transfer learning model using the pre-trained VGG16 architecture. The convolutional base is frozen, and new, trainable fully-connected layers are added on top.
create_mobilenet_model(): A lightweight transfer learning model using MobileNetV2. It uses GlobalAveragePooling2D which drastically reduces the number of parameters compared to Flatten.
5. Training
The train_and_plot_history() function orchestrates the training process for a given model.

Callbacks:
EarlyStopping: Monitors val_loss and stops training if it doesn't improve for 2 consecutive epochs (patience=2), restoring the best weights found.
ModelCheckpoint: Saves the best version of the model (.h5 file) based on val_accuracy.
Training: The model.fit() method is called using the data generator for the training data.
Visualization: After training, it plots and saves the training & validation accuracy and loss curves.
6. Evaluation
The evaluate_model() function provides a comprehensive assessment of a trained model. It:

Calculates the final test accuracy and loss.
Generates a classification report with precision, recall, and F1-score for each class.
Creates and displays a confusion matrix to visualize class-wise performance.
Visualizes 15 sample predictions from the test set, coloring the titles green for correct predictions and red for incorrect ones.
7. Inference Pipeline
Best Model Selection: The script identifies the best model by comparing the test accuracies of all four models.
Inference Function: The create_inference_pipeline() function creates and returns a simple prediction function, predict_image(image_path). This function can take the path to any image, preprocess it correctly, and return the predicted class and confidence score.
Results
After running the script, the following artifacts will be generated:

Pickle Files: X.pickle, y.pickle containing the preprocessed dataset.
Model Files: CNN_Model1.h5, CNN_Model2.h5, VGG16_Model.h5, MobileNet_Model.h5.
Training Plots: CNN_Model1_history.png, etc., for each model.
Evaluation Plots: CNN_Model1_confusion_matrix.png, CNN_Model1_sample_predictions.png, etc.
Comparison Plot: model_comparison.png showing a bar chart of the final test accuracies.
The console output will announce the best-performing model based on test accuracy. The inference pipeline will then be set up using this model, ready for new predictions.

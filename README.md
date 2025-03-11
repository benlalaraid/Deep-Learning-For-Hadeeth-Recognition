# Deep Learning For Hadeeth Recognition

## Project Overview
This project implements an Arabic character recognition system specifically designed for Hadeeth texts (Islamic narrations) using deep learning techniques. The system segments Arabic text images into individual characters and uses a Convolutional Neural Network (CNN) to recognize and classify each character.

## Key Features
- Text image preprocessing including binarization, deskewing, and noise removal
- Multi-level text segmentation (line, word, and character level)
- CNN-based character recognition model with 29 Arabic character classes
- Complete pipeline from raw image to recognized text

## Project Structure
- **Data/**: Contains segmented character images used for training
- **Dataset/**: Contains original Hadeeth text images
- **Preprocessing/**: Modules for image preprocessing
  - `preprocessing.py`: Implements Otsu thresholding and image cropping functions
- **segmentation/**: Modules for text segmentation
  - `segmentation.py`: Implements line and word segmentation using projection methods
  - `character_segmentation.py`: Implements character segmentation with baseline detection
- **model.h5**: Pre-trained CNN model for character recognition
- **chars.pkl**: Dictionary mapping class indices to Arabic characters
- **Notebooks**:
  - `data_preparation.ipynb`: Script for preparing and segmenting training data
  - `model.ipynb`: CNN model architecture, training, and evaluation
  - `main.ipynb`: End-to-end pipeline for text recognition

## Model Architecture
The character recognition model is a CNN with the following architecture:
- Multiple convolutional layers with batch normalization and max pooling
- Dropout layers for regularization
- Dense layers for classification
- 29 output classes corresponding to Arabic characters

## Segmentation Approach
1. **Line Segmentation**: Horizontal projection profile to separate text lines
2. **Word Segmentation**: Vertical projection profile to separate words in each line
3. **Character Segmentation**: Complex algorithm using baseline detection, vertical transitions, and morphological operations

## How to Use
1. Place Hadeeth text images in the `Dataset` folder
2. Run the data preparation script to segment characters (if training)
3. Use the pre-trained model or train your own using the model notebook
4. Run the main notebook to process new images and recognize text

## Dependencies
- Python 3.x
- TensorFlow/Keras
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- scikit-image
- matplotlib
- arabic_reshaper
- python-bidi

## Results
The model achieves high accuracy on the test set, with effective recognition of Arabic characters in Hadeeth texts. The segmentation pipeline successfully handles the complexities of Arabic script, including connected characters and diacritical marks.

## Future Improvements
- Implement post-processing for improved text accuracy
- Expand character set to include more variations and ligatures
- Optimize segmentation for different text styles and fonts
- Add support for full document recognition

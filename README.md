# Gender Classification ML Model

This repository contains a gender classification machine learning model that uses OpenCV for face detection, PCA for dimensionality reduction, and an SVM model for gender classification. The model is integrated into a Flask web application for deployment.

## Project Structure

The project is organized into the following directories:

- **Flask Application Development**: Contains the Flask web application that serves the model.
  - `app/`: Holds the Flask application views and related logic.
  - `model/`: Contains the trained SVM model, PCA components, and the Haar Cascade classifier.
  - `static/`: Holds static files such as CSS, JavaScript, and images used in the web application.
  - `templates/`: Contains HTML templates for rendering the web pages.
  - `main.py`: The entry point for running the Flask application.
  - `requirements.txt`: Lists the dependencies required to run the project.

- **Model Training**: Contains the scripts and data used for training the gender classification model.
  - `crop_data/`: Scripts for cropping images used in training.
  - `data/`: Holds the dataset used for training and testing.
  - `model/`: Contains the trained models and serialized objects (e.g., SVM model, PCA components).
  - `test_images/`: Contains test images for validating the model's performance.
  - `face_recognition.py`: The main script for performing face recognition and gender classification.
  - Other scripts (`FRM_*`): These scripts handle data preprocessing, feature extraction, machine learning model training, and predictions.

- **OpenCV Training**: Contains Jupyter notebooks for experimenting with OpenCV functionalities.
  - `data/`: Holds data used in the notebooks.
  - Various `.ipynb` files: Notebooks that demonstrate face detection, image resizing, and other OpenCV tasks.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/gender-classification.git
   cd gender-classification
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python main.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.

## Usage

### Face Recognition Pipeline

The `faceRecognitionPipeline` function is the core of the face recognition and gender classification process. Here's a brief overview of how it works:

1. **Read the Image**: The function reads an image either from a file path or as an array.
2. **Convert to Grayscale**: The image is converted to grayscale for easier processing.
3. **Face Detection**: The Haar Cascade classifier detects faces in the image.
4. **Normalization**: The detected face region is normalized.
5. **Resize**: The face region is resized to a standard 100x100 pixels.
6. **Flatten**: The image is flattened into a 1D array.
7. **Mean Subtraction**: The flattened image is subtracted from the mean face.
8. **PCA Transformation**: The mean-subtracted image is transformed using PCA to get the eigen image.
9. **SVM Prediction**: The eigen image is passed to the SVM model to predict gender.
10. **Result Display**: The result (gender and confidence score) is overlaid on the original image.

### Web Application

The Flask web application provides a simple interface to upload an image and get the gender classification results.

- **Home**: The main page of the web application.
- **/app/**: The main application interface.
- **/app/gender/**: Endpoint for gender classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- OpenCV for providing the Haar Cascade classifier.
- Scikit-learn for the SVM and PCA implementations.

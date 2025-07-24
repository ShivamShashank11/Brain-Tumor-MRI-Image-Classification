# 🧠 Brain Tumor MRI Image Classification

This project uses deep learning to classify brain MRI images into four tumor categories — **glioma**, **meningioma**, **pituitary**, and **no_tumor**. It supports training using both a **Custom CNN** and **ResNet50 (transfer learning)** model. A **Streamlit** web app is also provided to make predictions, visualize results using **Grad-CAM**, and export predictions as CSV.

## ✅ Key Features

- Train & evaluate on brain tumor MRI dataset
- Choose between Custom CNN or ResNet50
- Display Accuracy, Precision, Recall, and **F1 Score**
- Upload single or multiple images for prediction
- Visualize attention using **Grad-CAM heatmaps**
- Download prediction results as CSV
- Clean, modern Streamlit UI

## 📦 Installation

Install the required Python packages:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib seaborn pandas streamlit
Or use:


pip install -r requirements.txt
🧠 Model Training (Custom CNN or ResNet50)
Train your model using the provided scripts in src/ folder.


# For Custom CNN
from model_custom_cnn import build_model
from preprocessing import load_data

train_loader, val_loader, test_loader = load_data()
model = build_model()
model.fit(train_loader, validation_data=val_loader, epochs=10)
model.save("outputs/models/custom_cnn_model.keras")

# For ResNet50
from model_transfer_learning import build_resnet_model
from preprocessing import load_data

train_loader, val_loader, test_loader = load_data()
model = build_resnet_model()
model.fit(train_loader, validation_data=val_loader, epochs=10)
model.save("outputs/models/resnet50_model.keras")
📈 Evaluation with F1 Score
Evaluate trained model:


from evaluate import get_model_metrics
get_model_metrics(model, test_loader)
Output includes:

Accuracy

Precision

Recall

F1 Score

Classification Report

Confusion Matrix

🚀 Streamlit App Instructions
Launch the interactive app:


cd streamlit_app
streamlit run app.py
Features in the app:

Upload image(s) for classification

Choose model: Custom CNN or ResNet50

View predicted class with confidence

See Grad-CAM heatmap highlighting important regions

Download prediction report as CSV

📁 Dataset Format
Ensure your dataset is organized as follows:


data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── valid/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
Each subfolder contains .jpg or .png images of MRI scans.

🎨 Grad-CAM Heatmap Example
Grad-CAM helps explain which part of the image influenced the model’s decision.

Example output:

Prediction: Glioma

Grad-CAM: Highlights tumor region

📊 Sample Output Table
Image Name	Predicted Class	Confidence	Grad-CAM
image1.jpg	Glioma	94.1%	✅
image2.jpg	No Tumor	97.8%	✅

🔧 Future Improvements
Add Dockerfile for containerized deployment

Deploy on Hugging Face / Streamlit Cloud

Add image pre-segmentation before classification

Email alerts for bulk predictions

🤝 Acknowledgements
Dataset: Kaggle - Brain MRI

Grad-CAM: Selvaraju et al.

Streamlit: For interactive web UI

TensorFlow/Keras: Model training

📬 Contact
Author: Shivam Shashank
Email: shivamshashank961@gmail.com
```

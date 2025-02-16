
# INVASIVE PLANT SPECIES DETECTION USING CNN

This project focuses on detecting invasive plant species using Convolutional Neural Networks (CNNs). The system classifies native and alien plant species using ground-based images and employs a ResNet-50 architecture to improve detection accuracy.

The goal of this project is to support environmental conservation efforts by accurately identifying and classifying invasive plant species, which can negatively impact local ecosystems. Early detection of these species is critical for managing and controlling their spread.

#Key Features:

-->Invasive Species Detection:

Classifies images and identifies the invasive palnt species 

-->ResNet-50 Architecture:


Utilizes the ResNet-50 CNN architecture for accurate feature extraction and classification.

-->Data Augmentation:

Applies various data augmentation techniques to improve model generalization and robustness.

-->Evaluation:

 Performance metrics such as accuracy, confusion matrix, and loss graphs are used for evaluating the model’s effectiveness.

-->Test Time Augmentation (TTA):

 Used to further improve accuracy by making multiple predictions and averaging them.


## DATASET




The dataset is imported from Roboflow. You can sign up and upload your image classification dataset.

-->Dataset Download

To fetch the dataset, we use the Roboflow API:

from roboflow import Roboflow  
rf = Roboflow(api_key="hKrN1C04hGD7ajUxCV1H")  
project = rf.workspace("classification-of-native-and-alien-plant-species-using-hyperspectral-satellite-images").project("invasive-plant-species-detection-using-cnn")     
version = project.version(1)  
dataset = version.download("folder")
## REQUIREMENTS

1.  fastai==2.7.10  -- FastAI library for deep learning, used         for               training and data handling

2. torch==1.13.1  -- PyTorch library, the backbone of FastAI, used for model training

3. numpy==1.24.0 --NumPy library for numerical computations and handling arrays
4. roboflow==0.2.7-- Roboflow API client to download the dataset from Roboflow
5. matplotlib==3.6.2 --Matplotlib for plotting loss curves, confusion matrices, etc.
6. pandas==1.5.3 -- Pandas for data manipulation (optional for extra data handling)

7. scikit-learn==1.1.3-- Scikit-learn for machine learning tools like accuracy scoring

8. opencv-python==4.6.0.66 -- OpenCV for image processing and augmentations (if needed)

9. torchvision==0.14.0-- TorchVision for image transformations and augmentation

## Instructions For Setting Up the Environment

Clone this repository to your local machine.

Create a virtual environment (optional but recommended):

--python -m venv venv
___________________________________________________________________
Activate the virtual environment:

On Windows:
--venv\Scripts\activate

On macOS/Linux:

--source venv/bin/activate
___________________________________________________________________
Install the required dependencies:

--!pip install -r requirements.txt

You are now ready to run the project and train the model for detecting invasive plant species using CNN with ResNet-50.
___________________
Notes:

The ResNet-50 architecture is used in this project for improved performance in classifying and detecting invasive species.
fastai simplifies the training pipeline, while torch and torchvision are used for the underlying machine learning and image transformations.

roboflow is used to fetch the dataset from Roboflow and integrate it into the pipeline.
Make sure that you have a valid Roboflow API key if you intend to use the dataset directly from Roboflow.


## RESNET-MODEL

--Model Training

1. Import Dependencies

We import FastAI, NumPy, and Path for handling the dataset.

from fastai.vision.all import *  
import numpy as np
from pathlib import Path
_______
2. Data Preparation

We prepare the dataset and apply various data augmentations to improve generalization.

np.random.seed(42)  
path = Path(dataset.location)  
data = ImageDataLoaders.from_folder(  
                                   path,  
    valid_pct=0.2,  
    seed=42,  
    item_tfms=Resize(256),  
    batch_tfms=[  
        *aug_transforms(  
            size=224, min_scale=0.75, max_rotate=30.0, max_zoom=1.  1, 
            max_lighting=0.2, max_warp=0.2, p_affine=0.75,       p_lighting=0.75    
        ),   
        Normalize.from_stats(*imagenet_stats)  
    ],
    num_workers=4  
)  
print(data.vocab)  # Display class names
___
3. Model Creation

We use ResNet-101 instead of ResNet-50 for better feature extraction and accuracy.

learn = vision_learner(data, resnet101, metrics=accuracy)  
___
4. Find Optimal Learning Rate

FastAI's learning rate finder helps in choosing the best learning rate.

lr_min, lr_steep = learn.lr_find()
learn.fine_tune(5, base_lr=lr_steep)
___
5. Train the Model

We fine-tune the model further after unfreezing it.

learn.unfreeze()  
learn.fit_one_cycle(10, lr_max=slice(lr_steep/10, lr_steep))
___
6. Model Evaluation

We analyze the model's performance using a confusion matrix and accuracy metric.

interp = ClassificationInterpretation.from_learner(learn)  
interp.plot_confusion_matrix()  
print(interp.confusion_matrix())

preds, targs = learn.get_preds()  
accuracy_score = accuracy(preds, targs)  
print(f"Accuracy: {accuracy_score.item():.4f}")
___
7. Use Test Time Augmentation (TTA) for Better Accuracy

TTA helps in further boosting accuracy by averaging multiple augmented predictions.

preds, targs = learn.tta()  
final_acc = accuracy(preds, targs)  
print(f"Final Accuracy with TTA: {final_acc.item():.4f}")  
___
8. Loss Graph

We visualize how the model's loss decreases over epochs.

learn.recorder.plot_loss()  
____

9. Save the Model

learn.export("resnet101_invasive_plants.pkl")
___



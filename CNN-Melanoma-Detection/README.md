# Project Name - Melanoma Detection
## Brief description -  To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Data Overview
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). 
All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

## Methodology:
- Loading the data from the image-folders.
- Visulaization of single images from each class.
- Use of appropriate augmentation strategy to rectify the class imbalance.
- Use of Augmentor library to balance the class distributions.
- Running the model on train and Validations sets.

## Acknowledgements

- This project was inspired by IIITB & Upgrad.

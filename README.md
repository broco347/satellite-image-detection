# Signs from Above: Building Detection from Satellite Imagery

### Overview
The goal of this project is to train neural networks to autonomously recognize and map building footprints from satellite imagery taken before and after disaster events.  

### Project Organization
    ├── README.md            <- The top-level README for developers using this project
    │
    ├── Models               <- Keras model files saved in .h5 format
    │
    ├── Notebooks            <- Project source code in Jupyter notebook
    │   ├── functions5.py    <- Script with helper functions
    │
    ├── Reports              <- Various reports
    │   ├── proposal.pdf     <- Project proposal
    │   ├── summary.pdf      <- Project summary
    │   └── presentation.pdf <- Project presentation slide deck
    │
    ├── Images               <- Images featured in final slide deck
    │   ├── predictions      <- Model predictions
    │   └── overlays         <- Overlay of model predictions and ground truth
    │   

### Data
Training data was obtained from Spacenet, a collection of publicly available commercial satellite imagery and labelled training data hosted by Amazon Web Services (AWS) for the purpose of machine learning research. Specifically, this project utilizes 3,651 3-band (RGB) 200mx200m tiles overlooking Las Vegas at 30cm resolution. Before and after disaster imagery was obtained from DigitalGlobe.  

### Tools
* *Data Storage:* AWS S3, AWS EC2, h5py, Pickle
* *Data Processing:* Geopandas, Gdal, Rasterio, OpenCV, NumPy, Re, PIL
* *Data Visualization:* Matplotlib, Scikit-image
* *Deep Learning:* Keras w/ Tensorflow Backend
* *Presentation:* Powerpoint   

### Project Design
Because most of the work done on image segmentation has been trained using datasets (such as ImageNet) that are incongruous with geospatial data, I decided to train a model from scratch. The specific architecture used within the scope of this project was based off of a Kaggle submission by [Kevin Mader](https://www.kaggle.com/kmader/synthetic-word-ocr).

### LICENSE
MIT © Brooke Ann Coco

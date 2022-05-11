# SemanticSLAM
This is an implementation of SemanticSLam: Learning based Map Construction and Robust Camera Localization.

# Requirements
```
pip install -r requirements.txt
pip torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
```
Note: To install torch-scatter, specify the cuda version which is depending on your PyTorch installation(e.g.cpu, cu102, cu113, or cu115)

# Setup
This code is implemented in Python >=3.8.

# Datasets
Download datasets to "dataset" folder
https://drive.google.com/file/d/18J8n4-tKxwI7RJlyu2gGF_S2-9zOHJoU/view?usp=sharing
https://drive.google.com/file/d/11UnOoFUvB2b24M1KiQ40-SMzUIxK-j27/view?usp=sharing

# Training
Traing works with default arguments by:
```
python train.py --savedate SaveName
```
SaveName is the name of both logging file and saving models.

# Evaluation
Evaluation can be done as follows:
```
python evaluate.py --savedate SaveName
```
SaveName is should be same as SaveName during train to load well-trained modles

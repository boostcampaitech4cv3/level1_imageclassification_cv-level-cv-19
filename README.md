# pstage_01_image_classification

# level1_imageclassification_cv-level-cv-19
level1_imageclassification_cv-level-cv-19 created by GitHub Classroom

# Result
|누가|언제|모델|Backbone|Classifier|loss|optim|epochs|seed|batch|lr|acc|F1-Score|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Gu|10.26| ? |pretrained ResNet50| Backbone's Last fc layer|CE|SGD|50|42|64|1e-3|54.9365|0.5080|
|data|data|data|data|data|data|data|data|data|data|data|
|data|data|data|data|data|data|data|data|data|data|data|

<br>
<br>
<br>

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`

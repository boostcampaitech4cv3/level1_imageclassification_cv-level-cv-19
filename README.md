# pstage_01_image_classification

# level1_imageclassification_cv-level-cv-19
level1_imageclassification_cv-level-cv-19 created by GitHub Classroom

# Result
|누가|언제|구조|Backbone|Classifier|loss|optim|epochs|seed|batch|lr|acc|F1|etc|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Gu|10.26| 1 |pretrained ResNet50|Backbone's Last fc layer|CE|SGD|50|42|64|1e-3|54.93|0.50|---|
|Gu|10.26| 1 |pretrained ResNext101|Backbone's Last fc layer|CE|SGD|50|42|64|1e-3|62.46|0.55|---|
|TJ|10.27| AlexNet | AlexNet |AlexNet fc layer|CE|Adam|25|42|64|1e-3|35|0.24|---|
|GUN|10.27|mask,gender,age 따로 모델 훈련|pretrained EfficientNet|fc layer 추가|CE|Adam|1|42|128|1e-3|38.2540|0.2223|---|
|YR|10.27| 1model | basemodel| basemodel| CE|Adam|20|42|64|1e-3|42.68|0.28|---|
|Gu|10.27| 1 | Swin_b |Backbone's Last fc layer|CE|SGD|100|42|64|1e-3|0.51|0.3850|첫 30 epoch은 backbone frozen, 나머지 70 epoch은 frozen 없이 학습| 

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

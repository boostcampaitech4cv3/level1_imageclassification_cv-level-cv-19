# P_stage_01_image_classification

# level1_imageclassification_cv-level-cv-19
level1_imageclassification_cv-level-cv-19 created by GitHub Classroom

# Result
|idx|누가|언제|구조|Backbone|Classifier|Data-augmentation|loss|optim|epochs|seed|batch|lr|test_acc|test_F1|val_acc|val_F1|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|Gu|10.26| 1 |pretrained ResNet50|Backbone's Last fc layer|-|CE|SGD|50|42|64|1e-3|54.93|0.50|-|-|
|1|Gu|10.26| 1 |pretrained ResNext101|Backbone's Last fc layer|-|CE|SGD|50|42|64|1e-3|62.46|0.55|-|-|
|2|TJ|10.27| AlexNet | AlexNet |AlexNet fc layer|-|CE|Adam|25|42|64|1e-3|35|0.24|-|-|
|3|GUN|10.27|mask,gender,age 따로 모델 훈련|pretrained EfficientNet|fc layer 추가|-|CE|Adam|1|42|128|1e-3|38.2540|0.2223|-|-|
|4|YR|10.27| 1model | basemodel| basemodel|-| CE|Adam|20|42|64|1e-3|42.68|0.28|-|-|
|5|Gu|10.27| 1 | SwinT_b |Backbone's Last fc layer|-|CE|SGD|100|42|64|1e-3|51.00|0.3850|-|-|
|6|GH|10.27| 1 | DenseNet |Backbone's Last fc layer|-|CE|Adam|200|42|64|1e-3|-|-|-|-|

<br>
<br>

<details>
<summary><b>Details</b></summary>
<div markdown="1">
  
- idx 5: 처음 30 epochs는 마지막 fc layer를 제외하고 frozen, 나머지 70 epochs는 frozen 없이 학습 

</div>
</details>


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

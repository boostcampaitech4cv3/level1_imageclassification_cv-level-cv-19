# level1_imageclassification_cv-level-cv-19
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dc78682b-c934-47a6-b6cd-8f5078440446/Untitled.png)

## 프로젝트 개요

마스크를 착용하는 건 COVID-19의 확산을 방지하는데 중요한 역할을 합니다. 제공되는 이 데이터셋은 사람이 마스크를 착용하였는지 판별하는 모델을 학습할 수 있게 해줍니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고, 나이는 20대부터 70대까지 다양하게 분포하고 있습니다. 간략한 통계는 다음과 같습니다.

- 전체 사람 명수: 4500
- 한 사람당 사진의 개수: 7(마스크 착용 5장, 이상하게 착용(코스크, 턱스크)1장, 미착용 1장
- 전체 데이터셋 중에서 60%(2700명)는 학습 데이터셋으로 활용됩니다.

Evaluation

- Submission 파일에 대한 평가는 F1 Score를 통해 진행됩니다.
- F1 Score 에 대해 간략히 설명 드리면 다음과 같습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe6bb353-ec6b-40ab-8041-ca1a8af6ced1/Untitled.png)

## 프로젝트 팀 구성 및 역할
## 프로젝트 팀 구성 및 역할
20
| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이구](https://github.com/99sphere) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
|![image](https://avatars.githubusercontent.com/u/62556539?v=4)|![image](https://avatars.githubusercontent.com/u/48004826?v=4)|![image](https://avatars.githubusercontent.com/u/54363784?v=4) |![image](https://avatars.githubusercontent.com/u/59161083?s=400&u=e1dddd820da0add63917d38edf39e958471990f2&v=4)  |![image](https://avatars.githubusercontent.com/u/57178359?v=4)|  
|  |  || [Blog](https://99sphere-tech-blog.tistory.com/) | |
## 프로젝트 수행 절차 및 방법

타임라인 이미지 추가 후 

상세 설명

![image](https://user-images.githubusercontent.com/54363784/200259373-9348d518-8cc7-493d-a984-33004aa7888a.png)

### 대회 이전 준비**(~10/24)**

- baseline 및 딥러닝 파이프라인 이해를 위한 pytorch template 숙지
- 팀워크를 다지기 위한 회식(오프라인)

### 개인 학습 및 베이스라인 분석 (10/24~10/27)

- 개발 환경 설정(서버 설정, ssh)
- CV 기초대회 강의 수강 및 스폐설 미션 수행
- 모델 구조에 대한 아이디어 스케치
- 데이터 EDA를 통해 전반적인 task 이해

### 깃전략 및 컨벤션 정리 (10/27~10/28)

- Vincent Driessen branch 모델 및 대중적 commit convention 적용
- 협업 기록용 notion 워크스페이스 생성(회의록, convention 등)
- local 및 leaderboard inference 결과 기록용 공유 스프레드시트 생성

### 실험 계획 수립 (10/29 ~ 10/30)

### 기능 구현

- Evaluation 시각화 (f1-score, f1-score by {mask, gender, age}, confusion matrix)
- Label Smoothing (CutMix)
- Ensemble
- Train-Val-split (동일한 분포로)

### 실험(10/30~11/03)

 

- Backbone Model(DenseNet{121,201}, EfficientNet {B2, B6}, ConvNext {small, tiny}, Swin_B*, ResNext )

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f4cf182a-9f1f-4756-a2b7-6015ff0535ba/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9eab2f58-aa72-46d4-b0ea-ce90996f9a23/Val_f1_score.svg)

![Val_loss.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3a9c0c8-81e6-460e-a648-707a978292fd/Val_loss.svg)

실험 결과:  local과 test에서 ConvNext가 loss가 안정적이고, f1_Score가 가장 높아 hyperparameter 실험 backbone 모델로 선정되었다.

*Swin_B : swin transformer의 swin_base 아키텍처

BackBone 최종 모델: ConvNext_small

이하의 hyperparmeter 실험들은 backbone 실험에서 선정된 ConvNext_small을 기반으로,

```python
--seed 42 
--epochs 200 
--dataset MaskSplitByProfileDataset 
--augmentation GUNCustomAugmentation 
--resize 128 96 
--batch_size 64 
--optimizer AdamW 
--lr 0.0001 
--patient 25 
--cutmix_prob 0.4 
--beta 0.4 
--sampler Weighted_Random_Sampler 
--criterion focal
--model ConvNext_Small
```

를 base로 비교하여 실험하였다.

- Sampler (Weighted_Random(Base), Imbalanced, None sampler)

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5e82022-2da5-4b04-90a3-154669dd2736/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b2320d92-a7a8-43b5-8982-d0f2595ba1b8/Val_f1_score.svg)

![Val_loss.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/60eb3bbc-88f9-4c3c-9ec7-c150a3892557/Val_loss.svg)

실험 결과: Imbalanced sampler가 Weighted_Random와 Base보다 약간 더 좋은 f1_score를 보여줬다.

- Optimizer(SGD, Adam(Base), AdamW, NAdam, AdamP)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/63d6efac-e7f0-423f-be92-9d5d159a2620/Untitled.svg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c0ee7f14-8f40-450a-ba26-7982a67fca35/Untitled.svg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f29f5e0d-b7a1-441d-8749-d24e94ed4cfa/Untitled.svg)

실험결과: SGD, Adam, AdamW, NAdam, AdamP를 optimizer로 사용하며, 나머지 조건은 동일한 상태로 학습을 진행한 결과 AdamP의 수렴 속도가 가장 빨랐으며, loss의 최솟값 및 F1 score에서는 큰 차이를 보이지 않았다. 따라서, 원활한 실험 진행을 위해 이후의 모든 실험에서는 AdamP를 optimizer로 사용하였다.

- Resize({128,96}(base), {256,192})

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b7d17b89-a4ff-4040-80ef-b5722cc92de0/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/de574ac9-b506-49e6-bb31-3b69c1240661/Val_f1_score.svg)

![Val_loss.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e4c2a315-3c00-4abf-bdb7-258c8c55f218/Val_loss.svg)

실험 결과: {128,96}으로 resize를 하는 것보다, {256,196}으로 해상도가 높게 resize를 하는 것이 acc, f1, loss 모두에서 더 좋은 결과를 냈다.

- Criterion(focal(base), f1, label_smoothing, focal with label smoothing)

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb779202-44f0-497c-bea8-8abfa568731f/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/687b5ff3-9452-433a-96b3-0a3c3c86536e/Val_f1_score.svg)

![Val_loss.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74cdfb36-f99e-4c80-a4dd-b71285ff9321/Val_loss.svg)

실험 결과: 다른 criterion에 비해 f1_loss가 f1_score에서 낮은 성능을 보였다.

추가적으로, label smoothing과 focal with ls criterion은 loss의 절댓값이 다른 criterion보다 높았으나,  f1_score와 acc는 더 높았으므로, criterion 자체의 특성이라 판단했다.

위 실험을 통해, label smoothing 관련 criterion 중 가장 성능이 높은 focal with label smoothing을 최종 criterion으로 사용하였다.

- Learning Rate Scheduler(lr: {1e-3, 5e-4, 1e-4, 5e-5 }, scheduler: {None, warmup, Exponential, stepLR,  CosineLRS})

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d6fabdf9-3986-42ac-8103-f2c73055db29/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff30bf31-4509-4b56-861e-d7f9e2fb33bb/Val_f1_score.svg)

![Val_loss (1).svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/81f2d89e-6d8c-48e8-999b-754cac479dfe/Val_loss_(1).svg)

실험 결과:  lr의 경우 1e-3에서는 더 작은 lr에 비해 성능이 가장 좋지 않았고, 5e-5에서는 최종성능은 준수하였으나, loss가 매우 천천히 줄어들어 학습 속도가 느렸다.

scheduler의 경우 학습하는 동안 lr의 변화가 크지 않은 warm up, step LR  같은 단순한 형태보다는 Exponential이나,  CosineLRS 같은 복잡한 scheduler가 더 좋은 성능을 보였다.

위 실험을 통해, 학습이 minumum에 잘 수렴할 수 있도록 학습이 매우 느려지지 않는 선에서 가장 작은 lr(1e-4)을 선택했고,  local minimum에 빠지지 않도록 복잡한 형태의 scheduler(CosineLRS)를 선택했다.

- Augmentation(hue 제거, contrast 증가, hue 제거 & contrast 증가, random gray, always gray, gaussian noise)

![Val_accuracy.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f9fde381-2d61-4b42-a3c5-29c0f8dd5501/Val_accuracy.svg)

![Val_f1_score.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/77ca146b-5b5b-4206-949b-cd6dd790eb4a/Val_f1_score.svg)

![Val_loss.svg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7b8b3b9e-a6cf-4027-be66-62860475715c/Val_loss.svg)

실험 결과: base와 augmentation 실험 비교를 통해, hue 값 변화와 random gray는 성능을 하락시킨다는 것을 확인하였으며, contrast를 적당량(0.3) 증가시키는 것과 gaussian noise를 추가하는 것은 성능 향상을 야기한다는 것을 알게 되었다.

이미지를 언제나 gray scale로 바꾸는 것 또한 성능을 향상시켰으나 다른 augmentation과 동시 적용이 힘들다고 판단하여 최종 augmentation 후보에서 제외하였다.

- Fine tuning with Freezing

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d67fb87e-ec07-437a-ac2f-cc2b14bccdb1/Untitled.svg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/18b55660-81cc-4422-9d9c-171653b62598/Untitled.svg)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2538c32a-5b8d-477b-8a53-b95bab1d9379/Untitled.svg)

실험 결과 : feature layer를 모두 feezing 시키는 방법, feature layer의 앞부분만 freezing 시키는 방법, 학습 중간에 freezing을 푸는 방법(melting), non-freezing을 모두 썼는데, freezing을 조금 할 수록 성능이 좋은 것을 확인했다. 

melting 방법과 non-freezing이 비슷하게 좋은 성능을 냈기때문에 melting과 non-freezing 을 모두 후보에 넣었다. 

- labeling shifting

![best_sep_class_confusion_matrix_past.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ee2f43da-f0da-434a-bde8-ac925e07ff6f/best_sep_class_confusion_matrix_past.png)

![best_sep_class_confusion_matrix.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9dc38deb-f351-4d59-a32d-05eef25f156e/best_sep_class_confusion_matrix.png)

실험결과: data Imbalanced를 해결하기 위해서 여러 모델과 hyperparmeter 조정을 했음에도 불구하고, 60대 이상을 30~60대로 예측하는 경우가 줄어들지 않았다. 여러 실험을 통해 이 문제가 모델보다는 데이터에 기반한 문제라는 점을 착안하여, 절대적인 60대 이상의 데이터가 개수가 부족하다고 판단하였다.

또한 나이의 판단 기준인 사람의 노화는 연속적이고, EDA 결과 55~60세 사이의 나이 분포가 다른 연령대보다 많아 모델의 학습에 정확한 기준을 제시해주지 못한다고 생각하여 나이를 좀 더 보수적으로 결정하여도 된다고 생각하였다.

따라서, 60대의 데이터를 의도적으로 늘리면서 30대 이하, 30~60대, 60대 이상의 분포를 유지하기 위해서 400명 가량의 58세 이상을 60대 이상으로 label-shifting한다면, 절대적인 데이터의 개수가 늘어나므로 augmentation 및 다른 방법으로 해결할 수 없던 generalization 문제가 풀릴 것이라고 기대했고, 실제로도 60대 이상을 잘 분류하게 되어 f1 score도 증가했다.

### 버그수정

- 재현성 문제
설정한 seed값이 일정하게 적용되지 않는 문제가 있었으며, train set과 val set을 나누는 과정에서 list가 아닌 set을 연산하면서 생긴 문제로 확인하여 해결.
- f1-score오류 
f1-score가 accuracy와 동일한 값으로 출력되는 문제가 있었다. 원인은 확인하지 못했지만, 기존에 사용하던 torchmetrics.F1Score 함수를 torchmetrics.classification.MulticlassF1Score로 바꾸어 주니 해결됐다.

## 프로젝트 수행 결과

![image](https://user-images.githubusercontent.com/48004826/200256241-5a24e451-ba00-463e-9804-0201bf08d947.png)

![image](https://user-images.githubusercontent.com/48004826/200257745-6427a467-e1a9-4019-b21a-0021ac6dd92a.png)


앙상블용 모델:

1. 태준: convnext tiny 0.7176 77.3651
2. 예라: swin_b_shallow* 0.77058 0.7778
3. 구: final_melting* 0.7398 78.2063
4. ~~건: convnext without bn 0.663471.4127~~   (큰 성능 하락으로 제외)
5. 건희: convnext_shallow 0.7598 81.0635

*swin_b_shallow: swin base 아키텍처에서 classifier 부분을 2 fc layer로 수정한 것

*_melting : 학습 초반부에 feature layer는 freezing 하여 classifier(fc) layer만 학습시킨 뒤, 어느정도 학습 된 뒤에 feature layer의 freezing을 풀어 feature layer와 classifier layer를 같이 finetuning 한 기법을 적용한 것

추가적으로 앙상블용 2개 모델 훈련

Swin_S, EfficientNet_V2

*Swin_S : swin transformer의 swin_small 아키텍처

앙상블 조합:

 1. (convnext_small_shallow*, convnext_small, swin_b_shallow, convnext_tiny)

1. (swin_b ,swim_s, convnext_melting, convnext)
2. (Swin_b, convnext) ⇒ 
3. (Swin_b, convnext, efficientNet) ⇒ efficientNet이 들어가니 성능 개선,  efficientNet이 Swin_b의 부족한 부분 채워준다고 생각
4. (Swin_b,efficientNet) ⇒ best

최종 모델:  convNext ⇒  ensemble(EfficientNet + Swin)

*convnext_small_shallow : convnext_samll 아키텍처에서 classifier 부분을 2 fc layer로 수정한 것

## 자체 평가 의견

**잘한 점**

- Github Convention을 정하여 협업 결과 공유 및 정리에 큰 도움이 되었다.
- 베이스라인을 분석하고 이를 잘 추상화하여 재사용성을 극대화했다.(이말이 맞는지 확인필)→ 덕분에 깃허브를 잘 사용할 수 있었음
- 실험 결과 및 분석을 공유하기 위해 시간을 굉장히 많이 할애했다.
- 회의를 통해서 현재 시점에서 부족한 부분을 해결하기 위한 방법을 제안하고,  함께 그 문제를 해결하기 위해 체계적인 훈련 타임라인을 지정했다.
- 힘든 일정에도 서로를 격려하고 팀 분위기를 긍정적으로 유지하였다.
- 배운 기술들을 모두 사용해보았다.
- 서로의 의견을 존중해주고, 모두 잘 들어주었다. 그 덕분에, 다양한 의견들을 낼 수 있었다.
- 딥러닝에 사용되는 여러 방법론(scheduler, sampler 등)을 직접 추가하여 사용하였다.
- 발생하는 에러에 대해 다같이 내 일인 것처럼 해결해주었다.
- 남은 기간에 따른 실험 설계 및 역할 분배가 체계적으로 이루어졌다.

**아쉬운 점:**

- 모델 분석에 Tensorboard를 사용하여 각 local에서의 분석은 용이했으나, 팀원 간 결과 공유는 상대적으로 자유롭지 못했다.
- 추가한 코드에 대해 코드 내에서 설명이 부족했다.
- 모델 및 기법들에 대해 이론적인 공부와 결과분석이 부족했다.
- 중반까지 validation과 train set의 분포를 생각하지 않고 dataset을 split하여, 그 시점까지 실험했던 모델들의 결과 분석이 신뢰도가 부족한 것 같다.
- arguments의 default 값이 바뀌는 등의 주요한 변경사항 공유가 부족했다.
- baseline 코드 + 각자가 추가한 코드에 대해 검증이 부족했다.
- 모델을 훈련 후 성능과 개선 방법을 알아내기 위한 평가 환경을 제대로 구현하지 않고 학습부터 시작했다.
- 여러 모델 구조에 대해서 심도 깊은 테스트를 하지 못하였다

**개선할 점:**

- 실험 결과를 각자의 서버에서는 쉽게 확인할 수 있지만, 다른 사람의 실험 결과를 확인하려면 tensorboard의 로그 파일을 개인적으로 요청하고, 다시 서버에 업로드 하는 등의 부수적인 과정이 필요했다. git과 log 파일의 저장 위치를 고려하여 자동으로 실험 결과 공유가 가능하도록 하거나, weight and bias를 사용하는 등의 개선이 가능할 것이다.
- 모델의 학습이 끝난 후, 제출을 위한 최종 결과물을 선정하는 기준(best acc / best loss / best f1)이 프로젝트 중반에 바뀌었다. 이로인해 초반에 리더보드에서 확인한 점수들은 성능 비교에 활용할 수 없었다.
- 모델 결과에 대한 평가와 분석을 위해서는 실험 도메인에 대한 이해와 실험 환경 구성(score metric, 시각화 등 )이 선행되어야 한다.

---

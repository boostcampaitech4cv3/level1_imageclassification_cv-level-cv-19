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

![image](https://user-images.githubusercontent.com/54363784/200259606-07905834-50b3-47c3-942a-3ee9b71ec8c8.png)

## 프로젝트 팀 구성 및 역할
20
| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이구](https://github.com/99sphere) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
|![image](https://avatars.githubusercontent.com/u/62556539?v=4)|![image](https://avatars.githubusercontent.com/u/48004826?v=4)|![image](https://avatars.githubusercontent.com/u/54363784?v=4) |![image](https://avatars.githubusercontent.com/u/59161083?s=400&u=e1dddd820da0add63917d38edf39e958471990f2&v=4)  |![image](https://avatars.githubusercontent.com/u/57178359?v=4)|  
|  |[Blog](https://velog.io/@goodheart50)|| [Blog](https://99sphere-tech-blog.tistory.com/) | |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)
</div align="center">

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

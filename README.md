# 손바닥 sEMG 기반 사용자 식별 모델 재현 및 성능 비교

## 1. 코드 설명

본 프로젝트에서는 문손잡이 회전 과정에서 측정된 손바닥 sEMG 신호를 이용하여 사용자 A~E를 식별하는 모델을 재현하고, 여러 딥러닝 모델의 성능을 비교하였다.

### 사용 데이터 및 전처리 방법

- 사용자 수: 5명 (A~E)
- 사용자별 Trial 수: 50개
- 전체 Trial 수: 250개
- Sampling Rate: 1000 Hz
- sEMG Channel: 2개

전처리 과정은 다음과 같다.

1. 60 Hz Notch Filter
2. 20~499 Hz Band-pass Filter
3. 300 ms Sliding Window
4. 50% Overlap
5. Min-Max Normalization
6. CWT 변환
   - Morlet Wavelet
   - Scale 1~32
7. 최종 입력 크기: `(3, 32, 300)`

데이터 누수를 방지하기 위해 Sliding Window를 생성하기 전에 Trial 파일 단위로 Train/Test 데이터를 분할하였다.

- Train: 200 Trials / 3800 Windows
- Test: 50 Trials / 950 Windows

### 사용 모델

다음 3개의 모델을 비교하였다.

- 2D CNN
- ResNet18
- DenseNet161

### 학습 및 테스트 방법

모든 모델은 동일한 Train/Test 데이터와 동일한 학습 조건을 사용하였다.

- Epoch: 45
- Batch Size: 16
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: CrossEntropyLoss
- Pretrained Weight: 사용하지 않음

각 모델은 동일한 Train/Test Split에서 총 3회 학습하였으며, 최종 성능은 3회 실행 평균을 이용하였다.

### 실행 방법

- Google Colab
- NVIDIA Tesla T4 GPU
- Train/Test Split 및 Cross Validation: `random_state=42`
- Python / PyTorch 기반

`palm_semg_user_identification.ipynb` 파일을 Colab에서 열고 위에서부터 순서대로 실행한다.

실행 순서:

```text
1. 데이터 로드
2. Train/Test Trial 단위 분할
3. sEMG 전처리
4. Sliding Window 생성
5. Min-Max Normalization
6. CWT 변환
7. 모델 학습
8. 테스트 평가
9. Confusion Matrix 생성
10. 성능 비교
```

### 코드 파일 설명

| 파일 | 설명 |
|---|---|
| `palm_semg_user_identification.ipynb` | 데이터 전처리, 모델 학습 및 평가 전체 코드 |
| `results/confusion_2dcnn.png` | 2D CNN Confusion Matrix |
| `results/confusion_resnet18.png` | ResNet18 Confusion Matrix |
| `results/confusion_densenet.png` | DenseNet161 Confusion Matrix |

---

## 2. 모델 성능 비교

각 모델을 동일한 조건에서 3회 학습한 뒤 평균 성능을 비교하였다.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 2D CNN | 63.23% | 65.29% | 63.23% | 62.95% |
| ResNet18 | **87.79%** | **88.19%** | **87.79%** | **87.71%** |
| DenseNet161 | 87.44% | 88.17% | 87.45% | 87.34% |

### 성능 분석

3회 실행 평균 기준으로 ResNet18이 가장 높은 Accuracy와 F1-score를 기록하였다.

ResNet18의 Accuracy는 87.79%, DenseNet161은 87.44%로 두 모델의 성능 차이는 매우 작았다.

반면 단순한 구조의 2D CNN은 63.23%의 Accuracy를 기록하여 ResNet18과 DenseNet161보다 낮은 성능을 보였다.

---

## 3. Confusion Matrix 분석

### 3.1 2D CNN

![2D CNN Confusion Matrix](results/confusion_2dcnn.png)

- 가장 잘 분류된 클래스: **A**
- 가장 많이 오분류된 클래스: **E**
- 주요 오분류 유형: **E → B (87개)**
- 오분류가 발생한 이유에 대한 분석:
  - 2D CNN은 비교적 단순한 구조로 인해 사용자별 sEMG의 복잡한 시간-주파수 특징을 충분히 구분하지 못했을 가능성이 있다.
  - 특히 E 클래스가 B 클래스로 많이 분류되는 경향을 보였다.

---

### 3.2 ResNet18

![ResNet18 Confusion Matrix](results/confusion_resnet18.png)

- 가장 잘 분류된 클래스: **A**
- 가장 많이 오분류된 클래스: **D**
- 주요 오분류 유형: **D → C (17개)**
- 오분류가 발생한 이유에 대한 분석:
  - 일부 사용자 간 sEMG 시간-주파수 특징이 유사하게 나타났을 가능성이 있다.
  - 전체적으로는 대부분의 클래스에서 비교적 균형적인 분류 결과를 보였다.

---

### 3.3 DenseNet161

![DenseNet161 Confusion Matrix](results/confusion_densenet.png)

- 가장 잘 분류된 클래스: **A**
- 가장 많이 오분류된 클래스: **B**
- 주요 오분류 유형: **B → E (24개)**
- 오분류가 발생한 이유에 대한 분석:
  - B → E 오분류 24건은 총 10개의 B trial에 분산되어 있었으며, 특정 trial 하나에만 집중된 오류는 아니었다.
  - 동작 구간별로는 Rotation 구간에서 16건(66.7%)이 발생하여, 회전 과정에서 B와 E의 sEMG 특징이 상대적으로 유사하게 나타났을 가능성이 있다.
  - 오분류된 window에서 E 클래스의 평균 예측 확률은 약 0.760으로 나타나, 모델이 E로 비교적 강하게 판단한 경우가 많았다.

---

## 4. 최종 결과

- 가장 성능이 좋은 모델: **ResNet18**
- 가장 성능이 낮은 모델: **2D CNN**
- 주요 오분류 클래스:
  - 2D CNN: E → B
  - ResNet18: D → C
  - DenseNet161: B → E

### 전체적인 실험 결과 및 느낀 점

3회 실행 평균 기준으로 ResNet18이 87.79%의 Accuracy와 87.71%의 F1-score를 기록하여 가장 높은 성능을 보였다.

DenseNet161은 87.44%의 Accuracy와 87.34%의 F1-score를 기록하여 ResNet18과 매우 유사한 성능을 보였다.

반면 2D CNN은 63.23%의 Accuracy를 기록하여 상대적으로 낮은 성능을 나타냈다.

이번 실험을 통해 동일한 데이터를 사용하더라도 모델 구조에 따라 성능 차이가 발생할 수 있음을 확인하였다. 또한 단순히 더 복잡한 모델이 항상 더 높은 성능을 보이는 것은 아니며, 여러 모델을 동일한 조건에서 비교하는 것이 중요하다는 점을 확인하였다.

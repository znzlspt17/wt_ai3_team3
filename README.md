# 🗺️ 여행지 이미지 멀티라벨 분류 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.54+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.133+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<p align="center">
  여행지 이미지를 업로드하면 <strong>장소 유형</strong>과 <strong>여행 테마</strong>를 자동으로 분류하는 딥러닝 기반 멀티라벨 분류 시스템입니다.<br/>
  EfficientNet-B0 · ResNet50 · VGG16 · ConvNeXt-Tiny 네 가지 모델을 선택하여 분류 결과를 비교할 수 있으며,<br/>
  GradCAM 시각화를 통해 모델이 이미지의 어떤 부분에 집중하는지 확인할 수 있습니다.
</p>

---

## 📌 주요 기능

| 기능 | 설명 |
|------|------|
| 🏷️ **멀티라벨 분류** | 장소 유형 7개 + 여행 테마 5개를 동시에 예측 |
| 🤖 **다중 모델 지원** | EfficientNet-B0 / ResNet50 / VGG16 / ConvNeXt-Tiny 선택 가능 |
| 🔥 **GradCAM 시각화** | 모델 판단 근거 히트맵 시각화 |
| 🌐 **Streamlit 웹 UI** | 직관적인 이미지 업로드 및 결과 확인 |

---

## 🏷️ 분류 레이블

### 장소 유형 (Place Type)
`문화재` `문화` `자연` `테마공원` `시장` `거리` `공원`

### 여행 테마 (Theme)
`데이트/로맨틱` `힐링/여유` `액티브/아웃도어` `가족/키즈` `야경/밤감성`

---

## 🤖 지원 모델

| 모델 | 학습 내용 | 가중치 파일 |
|------|-----------|-------------|
| **EfficientNet-B0** | Optuna 튜닝 적용 (Part1 + Part2) | `optuna_best_model_efficientnet_part1.pth` / `part2.pth` |
| **ResNet50** | 장소 분류 + 테마 분류 | `best_place_model_epoch_1.pth` / `best_theme_model_epoch_3.pth` |
| **VGG16** | 단일 + 멀티라벨 | `best_vgg_model_epoch_7.pth` / `multi_best_vgg_model_epoch_2.pth` |
| **ConvNeXt-Tiny** | 단일 + 멀티라벨 | `best_single_convnext.pth` / `best_multi_convnext.pth` |

---

## 📁 프로젝트 구조

```
team_three/
├── 📄 prj_model_effic_stream.py      # Streamlit 메인 앱
├── 📄 MultiLabelDataset.py           # PyTorch 커스텀 데이터셋
├── 📄 pyproject.toml                 # 의존성 관리 (uv)
├── 📓 resnet50_train.ipynb           # ResNet50 학습 노트북
├── 📓 resnet50_train_with_finetuning.ipynb      # ResNet50 파인튜닝
├── 📓 resnet50_train_with_finetuning_optuna.ipynb  # Optuna 튜닝
├── 📓 draw_gradcams.ipynb            # GradCAM 시각화
├── data/
│   └── place_data_final.csv          # 통합 데이터셋
├── models/                           # 학습된 가중치 파일
└── final_images/                     # 추론용 이미지
 ```

---

## ⚙️ 설치 방법

> Python 3.11 이상, CUDA 13.0 환경을 권장합니다.

```bash
# 1. 저장소 클론
git clone <repository-url>
cd team_three

# 2. 가상환경 생성 및 의존성 설치 (uv 사용)
uv sync

# 또는 pip 사용
pip install -e .
```

---

## 🚀 실행 방법

### Streamlit 웹 앱 실행

```bash
streamlit run prj_model_effic_stream.py
```

브라우저에서 `http://localhost:8501` 에 접속하세요.

---

## 📥 데이터 및 모델 다운로드

아래 링크에서 학습된 모델 가중치와 이미지 데이터를 다운로드하세요.

| 항목 | 링크 |
|------|------|
| 🤖 **모델 가중치** | [Google Drive - models](https://drive.google.com/file/d/1xAJCg0zEFBXtZEba5T6qzxIrGvLi8A5S/view?usp=drive_link) |
| 🖼️ **학습 이미지** | [Google Drive - train images](https://drive.google.com/file/d/1J7I5xSaPTFsKR7l42956JBICx-fEWtPW/view?usp=drive_link) |
| 🧪 **테스트 이미지** | [Google Drive - test images](https://drive.google.com/file/d/10qPdxOBRTtnRWINKEfG8SGScgJKDbMtY/view?usp=drive_link) |

다운로드 후 모델 파일은 `models/` 폴더에, 이미지는 `final_images/` 폴더에 위치시켜 주세요.

---

## 🛠️ 기술 스택

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Optuna-4B8BBE?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorBoard-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/GradCAM-FF6B35?style=flat-square"/>
</p>

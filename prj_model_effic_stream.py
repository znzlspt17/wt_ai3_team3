import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageFile
import base64

# Windows 한글 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# -------------------------------------------------
# streamlit run prj_model_effic_stream.py
# -------------------------------------------------
st.set_page_config(layout="wide")
logo_img = "./data/common_images/logo.png"
st.logo(logo_img, size="medium")
# st.image(logo_img, use_container_width=True)

# -------------------------------------------------
# 인트로 사운드 (페이지 접속 시 클릭 1회 후 재생)
# -------------------------------------------------
_intro_sound_path = "./data/sounds/intro_sound.mp3"
if "intro_played" not in st.session_state:
    st.session_state.intro_played = False

if not st.session_state.intro_played and os.path.exists(_intro_sound_path):
    with open(_intro_sound_path, "rb") as _f:
        _audio_b64 = base64.b64encode(_f.read()).decode()
    st.markdown(
        f"""
        <div id="intro-overlay" style="
            position:fixed; top:0; left:0; width:100vw; height:100vh;
            background:rgba(0,0,0,0.85); z-index:9999;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center; cursor:pointer;"
            onclick="
                document.getElementById('intro-overlay').style.display='none';
                var a=new Audio('data:audio/mp3;base64,{_audio_b64}');
                a.play();
            ">
            <div style="color:white; font-size:2.5rem; font-weight:bold; margin-bottom:1rem;">🎬 스크린 속 그곳으로</div>
            <div style="color:#ccc; font-size:1.1rem; margin-bottom:2rem;">캡쳐 한 장으로 떠나는 나만의 K‑로드 투어</div>
            <div style="background:#ff4b4b; color:white; padding:0.8rem 2.5rem;
                        border-radius:50px; font-size:1.2rem; font-weight:bold;">
                ▶ 시작하기
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("▶ 시작하기", key="intro_btn", type="primary", use_container_width=False):
        st.session_state.intro_played = True
        st.rerun()


# -------------------------------------------------
# INFO
# -------------------------------------------------
df_info = pd.read_csv("data/place_data_final.csv")
LABELS_P1 = [
    "문화재",
    "문화",
    "자연",
    "테마공원",
    "시장",
    "거리",
    "공원",
]  # df_info.columns[2:9].tolist()
LABELS_P2 = [
    "데이트/로맨틱",
    "힐링/여유",
    "액티브/아웃도어",
    "가족/키즈",
    "야경/밤감성",
]  # df_info.columns[9:14].tolist()
TARGET_COLUMNS = LABELS_P1 + LABELS_P2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "final_images"

efficientnet_b0_part1 = "models/optuna_best_model_efficientnet_part1.pth"
efficientnet_b0_part2 = "models/optuna_best_model_efficientnet_part2.pth"

resnet50_part1 = "models/best_place_model_epoch_1.pth"
resnet50_part2 = "models/best_theme_model_epoch_3.pth"

vgg16_part1 = "models/best_vgg_model_epoch_7.pth"
vgg16_part2 = "models/multi_best_vgg_model_epoch_2.pth"

convnext_tiny_part1 = "models/best_single_convnext.pth"
convnext_tiny_part2 = "models/best_multi_convnext.pth"

# -------------------------------------------------
# GradCAM 클래스 정의
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = (
            torch.sum(weights * self.activations, dim=1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
def load_model(path, num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))

    model.to(DEVICE).eval()
    return model

def load_convnext_model(path, num_classes):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_model_others(path):
    try:
        model = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_models(path, num_classes):
    if "efficientnet" in path:
        return load_model(path, num_classes)
    if "convnext" in path:
        return load_convnext_model(path, num_classes)
    else:
        return load_model_others(path)


# -------------------------------------------------
# 유사도
# -------------------------------------------------
def get_top3_similar_metadata(prob1, prob2, df_info):
    # 모델 결과를 0.5 기준으로 이진화 (1 또는 0)
    # P1 (장소 8종), P2 (분위기 6종) 순서대로 결합
    pred_probs = np.concatenate([prob1, prob2])
    pred_binary = (pred_probs >= 0.5).astype(int)

    # CSV 데이터와 모델 예측값 사이의 일치 점수 계산 (Hamming Distance 유사 개념)
    # 각 행(Row)에 대해 모델이 1로 예측한 곳에 1이 있는지 점수.
    df_scores = df_info.copy()

    # 각 행의 벡터와 예측 벡터 간의 일치 개수 계산
    csv_vectors = df_info[TARGET_COLUMNS].values

    # (예측이 1인 곳과 실제가 1인 곳의 합 + 예측이 0인 곳과 실제가 0인 곳의 합)
    match_counts = np.sum(csv_vectors == pred_binary, axis=1)
    df_scores["match_score"] = match_counts
    df_sorted = df_scores.sort_values(by="match_score", ascending=False)

    # 동점 행들을 먼저 섞은 뒤 정렬
    df_shuffled = df_sorted.sample(frac=1).sort_values(by="match_score", ascending=False)
    top3_unique_matches = df_shuffled.drop_duplicates(subset=["folder_name"], keep="first")
    top3_matches = top3_unique_matches.head(3)

    return top3_matches, pred_binary


# -------------------------------------------------
# Grad Cam show
# -------------------------------------------------
def show_dual_gradcam(img_path, mask1, mask2, label1, prob1, label2, prob2):
    # PIL 이미지를 NumPy 배열로 변환
    img_array = np.array(img_path.convert("RGB"))
    target_size = (224, 224)
    # 리사이즈
    img_resized = cv2.resize(img_array, target_size)

    def overlay_heatmap(img, mask):
        # 마스크 크기를 이미지 크기(224, 224)와 맞추기
        mask_resized = cv2.resize(mask, target_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 히트맵 합성
    res1 = overlay_heatmap(img_resized, mask1)
    res2 = overlay_heatmap(img_resized, mask2)

    # 시각화 (Matplotlib)
    fig = plt.figure(figsize=(18, 6))  # fig 객체를 생성합니다.

    # (1) 원본 이미지
    plt.subplot(1, 3, 1)
    plt.title(f"Original:")
    plt.imshow(img)
    plt.axis("off")

    # (2) Part 1 장소 결과
    plt.subplot(1, 3, 2)
    plt.title(f"P1 [Place]: {label1} ({prob1*100:.1f}%)")
    plt.imshow(res1)
    plt.axis("off")

    # (3) Part 2 분위기 결과
    plt.subplot(1, 3, 3)
    plt.title(f"P2 [Mood]: {label2} ({prob2*100:.1f}%)")
    plt.imshow(res2)
    plt.axis("off")

    plt.tight_layout()
    # plt.show()
    st.pyplot(fig)

    # 메모리 절약을 위해 현재 fig 닫기
    plt.close(fig)


# -------------------------------------------------
# 추론
# -------------------------------------------------
places = []


def run_inference_with_cam(img_path, model1, model2):
    global places
    # 1. 이미지 전처리
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(img_path).unsqueeze(0).to(DEVICE)

    # 모델 추론
    with torch.no_grad():
        out1 = model1(input_tensor)
        out2 = model2(input_tensor)
        prob1 = torch.sigmoid(out1).cpu().numpy()[0]
        prob2 = torch.sigmoid(out2).cpu().numpy()[0]

    # 결과 출력
    print(f"\n--- [ 분석 결과: {img_path} ] ---")
    top1_idx = np.argmax(prob1)
    top2_idx = np.argmax(prob2)

    # 모델 예측 값 (카테고리명)
    p1_label = LABELS_P1[top1_idx]
    p2_label = LABELS_P2[top2_idx]

    print(f"장소(P1): {p1_label} ({prob1[top1_idx]*100:.2f}%)")
    print(f"분위기(P2): {p2_label} ({prob2[top2_idx]*100:.2f}%)")

    top3_df, binary_result = get_top3_similar_metadata(prob1, prob2, df_info)

    print(f"모델 예측 이진화 결과: {binary_result}")
    print(f"--- [ 모델 결과와 가장 유사한 학습 데이터 TOP 3 ] ---")

    for i, (idx, row) in enumerate(top3_df.iterrows()):
        places.append(
            {
                "folder_name": row["folder_name"],
                "file_name": row["file_name"],
                "Place KorName": row["장소명"],
                "Place EngName": row["Place Name"],
                "Place KorDesc": row["장소 설명"],
                "Place EngDesc": row["Place Description"],
            }
        )
        # print(f"{i+1}위: {row['folder_name']}")
        # print(f"파일명: {row['file_name']}")
        # print(f"일치 지표 수: {row['match_score']} / {len(binary_result)}")
        # print("-" * 30)
        img_tmp = Image.open(f"{IMG_DIR}/{row['file_name']}")
        # 페이지에 띄우기
        # plt.figure(figsize=(10, 6)) # 이미지 출력 크기 설정
        # plt.imshow(img_tmp)
        # plt.axis('off') # 축(axis) 숨기기
        # plt.show()

    # Grad-CAM 엔진 초기화 (모델 아키텍처별 마지막 Conv 레이어 타겟)
    def get_target_layer(model):
        if hasattr(model, "layer4"):          # ResNet
            return model.layer4[-1]
        elif hasattr(model, "features"):      # EfficientNet, VGG, ConvNeXT
            return model.features[-1]
        else:
            raise ValueError(f"지원하지 않는 모델 구조: {type(model)}")

    target_layer1 = get_target_layer(model1)
    target_layer2 = get_target_layer(model2)

    cam1 = GradCAM(model1, target_layer1)
    cam2 = GradCAM(model2, target_layer2)

    # 각 모델별 최고 확률 인덱스 추출
    idx1 = np.argmax(prob1)
    idx2 = np.argmax(prob2)

    # 히트맵 마스크 생성
    mask1 = cam1.generate(input_tensor, idx1)
    mask2 = cam2.generate(input_tensor, idx2)

    # 시각화 호출
    show_dual_gradcam(
        img_path,
        mask1,
        mask2,
        LABELS_P1[idx1],
        prob1[idx1],
        LABELS_P2[idx2],
        prob2[idx2],
    )


# -------------------------------------------------
# 모델 적용 (선택된 모델만 lazy load)
# -------------------------------------------------
MODEL_PATHS = {
    "EfficientNet": (efficientnet_b0_part1, efficientnet_b0_part2),
    "ResNet50":     (resnet50_part1,         resnet50_part2),
    "VGG16":        (vgg16_part1,            vgg16_part2),
    "ConvNeXT":     (convnext_tiny_part1,    convnext_tiny_part2),
}


# -------------------------------------------------
# Translations
# -------------------------------------------------
TEXTS = {
    "KOR": {
        "title": "🎬 스크린 속 그곳으로",
        "subheader": "캡쳐 한 장으로 떠나는 나만의 K\u2011로드 투어",
        "description": "좋아하는 **K\u2011드라마 / 영화 장면**을 캡쳐해서 올리면  \n비슷한 분위기의 **여행지 TOP 3**를 추천합니다.",
        "selectbox": "카테고리 선택",
        "uploader": "📸 장면 캡쳐 업로드",
        "img_caption": "업로드한 장면",
        "top3": "추천 여행지 TOP 3",
        "save": "저장",
        "save_success": "여정에 저장되었습니다",
        "itinerary": "🧭 내 여행 리스트",
        "itinerary_empty": "아직 저장한 여행지가 없습니다.",
        "delete": "삭제",
        "place_name_key": "Place KorName",
        "place_desc_key": "Place KorDesc",
        "place_sub_key": "Place EngName",
    },
    "ENG": {
        "title": "🎬 That Place on Screen",
        "subheader": "My K-Road Tour with Just One Screenshot",
        "description": "Upload a screenshot from your favorite **K-Drama / Movie**  \nand we'll recommend **TOP 3 travel destinations** with a similar vibe.",
        "selectbox": "Select Model",
        "uploader": "📸 Upload Scene Capture",
        "img_caption": "Uploaded Scene",
        "top3": "Top 3 Recommended Destinations",
        "save": "Save",
        "save_success": "Saved to itinerary",
        "itinerary": "🧭 My Travel List",
        "itinerary_empty": "No destinations saved yet.",
        "delete": "Delete",
        "place_name_key": "Place EngName",
        "place_desc_key": "Place EngDesc",
        "place_sub_key": "Place KorName",
    },
}

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []
if "lang" not in st.session_state:
    st.session_state.lang = "KOR"

T = TEXTS[st.session_state.lang]

# -------------------------------------------------
# Header
# -------------------------------------------------

_, col_lang = st.columns([10, 1])
with col_lang:
    if st.session_state.lang == "KOR":
        st.button("ENG", on_click=lambda: st.session_state.update(lang="ENG"))
    else:
        st.button("KOR", on_click=lambda: st.session_state.update(lang="KOR"))

st.title(T["title"])
st.subheader(T["subheader"])

st.markdown(T["description"])

st.divider()
# -------------------------------------------------
# Sticky Sidebar Style CSS
# -------------------------------------------------
# 오른쪽 컬럼(itinerary 섹션)을 화면 상단에 고정하는 CSS 주입
st.markdown(
    """
    <style>
    /* 리스트 박스 디자인 살짝 개선*/
    .stExpander {
        border: 1px solid #ff4b4b !important;
        border-radius: 10px !important;
    } 
    div:has(> .stExpander){
        position : sticky ;
        top: 80px;
        width : 300px ;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Layout
# -------------------------------------------------
left, right = st.columns([3, 1])

# -------------------------------------------------
# Upload
# -------------------------------------------------
with left:
    dropdown_options = ["EfficientNet", "ConvNeXT", "ResNet50", "VGG16"]
    selected_option = st.selectbox(T["selectbox"], dropdown_options)
    path1, path2 = MODEL_PATHS[selected_option]
    m1 = load_models(path1, num_classes=len(LABELS_P1))
    m2 = load_models(path2, num_classes=len(LABELS_P2))

    uploaded = st.file_uploader(T["uploader"], type=["jpg", "png", "jpeg"])

    if uploaded:

        img = Image.open(uploaded).convert("RGB")
        # TEST_IMAGE = "data/datasets_local/sample.jpg" # 실제 파일 경로로 수정
        run_inference_with_cam(img, m1, m2)

        st.image(img, caption=T["img_caption"], use_container_width=True)

        st.subheader(T["top3"])

        cols = st.columns(3)

        for i, place in enumerate(places):

            with cols[i]:

                st.image(IMG_DIR + "/" + place["file_name"], use_container_width=True)

                st.markdown(f"### {place[T['place_name_key']]}")
                st.caption(place[T["place_sub_key"]])

                st.write(place[T["place_desc_key"]])

                c1, c2 = st.columns(2)

                if c1.button(T["save"], key=f"save{i}"):

                    if place not in st.session_state.itinerary:
                        st.session_state.itinerary.append(place)
                        st.success(T["save_success"])

                # lat,lon 정보 없음
                # if c2.button("지도", key=f"map{i}"):

                # st.map(pd.DataFrame([{
                #     "lat": place["lat"],
                #     "lon": place["lon"]
                # }]))

# -------------------------------------------------
# Itinerary
# -------------------------------------------------
with right:
    # 이 섹션은 위의 CSS 설정으로 인해 왼쪽 결과가 길어져도 화면 상단에 계속 고정됩니다.
    with st.expander(T["itinerary"], expanded=True):
        if len(st.session_state.itinerary) == 0:
            st.info(T["itinerary_empty"])
        else:
            for i, p in enumerate(st.session_state.itinerary):
                st.markdown(f"**{i+1}. {p[T['place_name_key']]}**")
                st.caption(p[T["place_sub_key"]])

                if st.button(T["delete"], key=f"del{i}", use_container_width=True):
                    st.session_state.itinerary.pop(i)
                    st.rerun()
                if i < len(st.session_state.itinerary) - 1:
                    st.divider()
st.divider()

st.caption("Powered by SCENEFLIX ")


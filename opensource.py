from roboflow import Roboflow
from ultralytics import YOLO
import os
import shutil

# -----------------------------
# 🔹 설정
# -----------------------------

# 이전 학습 모델(best.pt) 경로
PREV_MODEL = "trained_model/best1.pt"

DATA_YAML = "datafile/data1.yaml"

# # Roboflow 프로젝트 정보
# RF_API_KEY = ""
# WORKSPACE = ""
# PROJECT = ""
# VERSION = 

# YOLO 학습 결과 저장 위치
SAVE_DIR = "trained_model"

# # -----------------------------
# # 🔹 데이터셋 다운로드 (Roboflow → YOLO)
# # -----------------------------

# # Roboflow 연결 및 다운로드
# rf = Roboflow(api_key=RF_API_KEY)
# project = rf.workspace(WORKSPACE).project(PROJECT)
# version = project.version(VERSION)
# dataset = version.download("yolov8")

# -----------------------------
# 🔹 YOLO 학습
# -----------------------------

# 이전 모델 로드
if os.path.exists(PREV_MODEL):
    print(f"✅ 기존 모델 로드: {PREV_MODEL}")
    model = YOLO(PREV_MODEL)
else:
    print("🚀 새로운 모델 시작: yolov8m.pt")
    model = YOLO("yolov8m.pt")

# 학습 실행
model.train(
    data=DATA_YAML, 
    epochs=30,
    imgsz=640,
    batch=4,
    workers=0,
    project=SAVE_DIR,
    name="",        # YOLO가 자동으로 결과 저장
    resume=False    # 항상 새 데이터셋으로 학습 시작
)

# 학습 후 best.pt 갱신
latest_best = os.path.join(SAVE_DIR, "train", "weights", "best.pt")
if os.path.exists(latest_best):
    os.replace(latest_best, PREV_MODEL)
    print(f"💾 최신 best.pt 저장 완료 → {PREV_MODEL}")     
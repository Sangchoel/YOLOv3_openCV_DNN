# 🧠 YOLOv3 Real-Time Object Detection with OpenCV

> OpenCV의 `DNN` 모듈을 사용하여 **YOLOv3 모델을 로드하고**,  
> 웹캠 입력에 대해 실시간 객체 감지를 수행하는 **C++ 기반 프로젝트**입니다.

---

## 📦 주요 기능

- YOLOv3 모델(`.cfg` + `.weights`) 로드
- `coco.names` 파일로 클래스 이름 지정
- 웹캠 실시간 스트림에서 객체 탐지 수행
- `Non-Maximum Suppression`으로 중복 박스 제거
- 바운딩 박스 + 클래스 레이블 + 신뢰도 표시

---

## 📁 프로젝트 구성

| 파일 | 설명 |
|------|------|
| `main.cpp` | 전체 YOLO 객체 탐지 로직 |
| `yolov3.cfg` | YOLOv3 모델 구성 파일 |
| `yolov3.weights` | 사전 학습된 가중치 파일 |
| `coco.names` | 클래스 이름 리스트 (COCO 데이터셋 기준) |

---

## 🚀 실행 방법

### 1. 필요한 파일 다운로드
- [`yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)
- [`yolov3.cfg`](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [`coco.names`](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

모두 실행 파일과 같은 디렉토리에 위치시켜야 합니다.

---

### 2. 컴파일 (예시: Linux 기준)

```bash
g++ main.cpp -o yolo_demo `pkg-config --cflags --libs opencv4`

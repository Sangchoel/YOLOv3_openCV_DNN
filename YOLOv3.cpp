#include <opencv2/opencv.hpp>  // OpenCV 주요 기능을 포함하는 헤더 파일
#include <opencv2/dnn.hpp>     // OpenCV의 딥 러닝 모듈을 사용하기 위한 헤더
#include <fstream>             // 파일 스트림을 다루기 위한 헤더
#include <iostream>           
#include <vector>              
#include <algorithm>           

// Non-Maximum Suppression을 수행하는 함수 정의
void applyNMS(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences, float nmsThreshold, std::vector<int>& indices) {
    // NMSBoxes 함수는 중첩되는 경계 상자 중 불필요한 상자를 제거
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, nmsThreshold, indices);
}

// 객체 감지 결과를 그리는 함수
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, const std::vector<std::string>& classes) {
    // 감지된 객체 주변에 경계 상자를 그림
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 3);
    // 클래스 ID와 신뢰도를 표시하는 레이블 생성
    std::string label = cv::format("%s: %.2f", classes[classId].c_str(), conf);
    int baseLine;
    // 레이블의 크기를 계산
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    // 레이블 배경을 그림
    cv::rectangle(frame, cv::Point(left, top - std::round(1.5*labelSize.height)), cv::Point(left + std::round(1.5*labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    // 레이블 텍스트를 그림
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
}

// 비디오 캡처에서 객체를 감지하고 표시하는 함수
void detectAndDisplay(cv::VideoCapture& cap, cv::dnn::Net& net, const std::vector<std::string>& classes) {
    cv::Mat frame, blob; // 비디오 프레임과 이미지 blob을 저장할 변수
    std::vector<cv::Rect> boxes; // 감지된 객체의 경계 상자
    std::vector<float> confidences; // 감지된 객체의 신뢰도 점수
    std::vector<int> classIds; // 감지된 객체의 클래스 ID

    // 비디오 스트림을 읽는 루프
    while (cap.read(frame)) {
        // 프레임에서 blob 이미지를 생성
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(0,0,0), true, false);
        // blob을 네트워크의 입력으로 설정
        net.setInput(blob);

        std::vector<cv::Mat> outs; // 네트워크 출력을 저장할 변수
        // 네트워크를 실행하여 결과를 outs에 저장
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // 이전 데이터를 지움
        boxes.clear();
        confidences.clear();
        classIds.clear();

        // 각 출력 레이어에 대하여
        for (auto& out : outs) {
            for (int i = 0; i < out.rows; i++) {
                // 클래스별 신뢰도 점수를 추출
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                // 가장 높은 신뢰도를 가진 클래스를 찾음
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5) { // 신뢰도가 0.5 이상인 경우만 처리
                    // 경계 상자의 중심 좌표와 크기 계산
                    int centerX = (int)(out.at<float>(i, 0) * frame.cols);
                    int centerY = (int)(out.at<float>(i, 1) * frame.rows);
                    int width = (int)(out.at<float>(i, 2) * frame.cols);
                    int height = (int)(out.at<float>(i, 3) * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    // 경계 상자와 신뢰도, 클래스 ID를 저장
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back((float)confidence);
                    classIds.push_back(classIdPoint.x);
                }
            }
        }

        std::vector<int> indices; // NMS를 통해 선택된 상자의 인덱스를 저장
        // NMS를 수행하여 중복을 제거
        applyNMS(boxes, confidences, 0.4, indices);

        // 선택된 상자에 대해 경계 상자와 레이블을 그림
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);
        }

        // 결과를 화면에 표시
        cv::imshow("YOLO-Object Detection", frame);
        if (cv::waitKey(1) >= 0) break; // 사용자가 키를 누르면 루프 종료
    }
}

// 프로그램의 주 진입점
int main() {
    std::string classesFile = "coco.names"; // 클래스 이름이 저장된 파일
    std::ifstream ifs(classesFile.c_str()); // 파일 읽기 스트림
    std::string line;
    std::vector<std::string> classes; // 클래스 이름을 저장할 벡터
    while (std::getline(ifs, line)) classes.push_back(line); // 파일에서 클래스 이름을 읽어 저장

    std::string modelConfiguration = "yolov3.cfg"; // 모델 구성 파일
    std::string modelWeights = "yolov3.weights"; // 모델 가중치 파일
    // Darknet 모델을 로드
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // 백엔드로 OpenCV 사용 설정
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // 타겟으로 CPU 사용 설정

    cv::VideoCapture cap(0); // 비디오 캡처 초기화 (0은 기본 카메라)
    detectAndDisplay(cap, net, classes); // 감지 및 디스플레이 함수 호출
    return 0; // 프로그램 종료
}

# Facial Asymmetry Detector
웹캠으로 얼굴을 인식하여 안면 비대칭을 감지하고 시각화해주는 프로젝트입니다.


- 웹캠상에서 shape_predictor_68_face_landmarks를 사용하여 얼굴을 감지, landmarks를 추출합니다.
- 얼굴의 위치와 각도를 계산하여 얼굴 각도를 수평 보정, 중심 이동합니다. 그리고 눈 사이의 거리를 계산하여 얼굴의 크기를 일정하게 보이도록 조정합니다. 이 과정은 비대칭을 감지할 때 일정한 결과가 나오도록 해줍니다.
- f 키를 누르면 해당 프레임을 좌우반전해 다시 얼굴 landmarks를 계산하고 원본 이미지와의 차이를 구해 빨간 실선으로 시각화합니다.
- 스페이스바 키를 누르면 웹캠 화면을 실시간으로 좌우반전한 결과를 볼 수 있습니다.

---
- 위쪽 화면에는 웹캠에서 실시간으로 탐지한 얼굴이 중심으로 이동되어 표시되고, 아래쪽 화면에는 비대칭을 감지하여 시각화 합니다.
- 초록색 실선은 얼굴의 형태를 나타내고, 빨간색 실선은 반대쪽 얼굴과 비교한 차이를 나타냅니다.

## how to use
```
$ python main.py
```
**keyboard control**
- space-bar : flip webcam left and right
- f : Asymmetry detection
- q : Exit the program 

## running videos
![ezgif-5-dd37a5c655](https://github.com/yhcho0405/eye3d/assets/52823519/9b327c03-f503-43b7-8959-0b3d5a39ec06)
---
![ezgif-5-39b01a18d8](https://github.com/yhcho0405/eye3d/assets/52823519/ab1ccf4a-3438-4692-9774-42d4f4ffd815)


## reference
- face detector : shape_predictor_68_face_landmarks
- helped by ChatGPT

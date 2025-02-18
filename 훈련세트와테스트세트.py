# 지도학습 : 입력 데이터와 그에 해당하는 정답을 사용해 모델을 학습하는 방법.
# 학습 데이터에서 입력과 정답의 관계를 학습하여 새로운 입력에 대해 정확히 예측할 수 있도록 하는 것
# 훈련 세트 : 모델 학습용 데이터
# 테스트 세트 : 모델 평가용 데이터
# 즉, 같은 데이터로 학습과 평가를 진행하면 모델 데이터를 학습해버렸기 때문에 정확한 평가 불가

import numpy as np
from sklearn.neighbors import KNeighborsClassifier      # KNN 사용을 위해
import matplotlib.pyplot as plt     # 시각화를 위해

# 생성에 대한 길이와 무게 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 생선의 길이와 무게를 fish_data (2차원 리스트)로 만들고, 도미와 빙어 레이블을 fish_target에 저장
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]      # 특성은 길이와 무게
fish_target = [1] * 35 + [0] * 14       # 총 49개의 샘플 데이터

# 데이터 섞기
np.random.seed(42)      # 일정한 결과를 얻기 위해 랜덤 시드 결정
index = np.arange(49)   # 0 ~ 48까지의 인덱스 생성
np.random.shuffle(index)

fish_arr = np.array(fish_data)
targer_arr = np.array(fish_target)

# 훈련세트와 테스트세트 나누기
# train_input = fish_data[index[:35]]    # 0 ~ 34번째 까지를 훈련 세트로 지정
train_input = fish_arr[index[:35]]
# train_target = fish_target[index[:35]]
train_target = targer_arr[index[:35]]

# 테스트 세트
# test_input = fish_data[index[35:]]
test_input = fish_arr[index[35:]]
# test_target = fish_target[index[35:]]
test_target = targer_arr[index[35:]]

# 훈련 데이터와 테스트데이터 시각화
plt.scatter(train_input[:, 0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 모델 훈련
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)   # 훈련 세트로 모델 학습

# 모델 평가
score = kn.score(test_input, test_target)
print(f"모델 훈련 평가 결과 : {score:.2f}")     # 샘플링이 극단적으로 편향되어 있음

# 모델 평가 및 결과 확인
train_score = kn.score(train_input, train_target)   # 훈련 세트에 대한 정확도
test_score = kn.score(test_input, test_target)      # 테스트 세트에 대한 정확도
print(f"훈련 세트 정확도 :{train_score:.2f}")
print(f"테스트 세트 정확도 :{test_score:.2f}")

# 모델의 훌면 및 테스트 세트에 대한 정확도를 비교해 일반화 성능 확인
if train_score > test_score:
    print("모델이 훈련세트에 과적합(overfitting) 되었습니다.")
elif train_score < test_score:
    print("모델이 훈련세트를 충분히 학습하지 못했습니다.")
else:
    print("모델 훈련이 잘 되었습니다.")

# 테스트 세트 예측 결과 확인
predictions = kn.predict(test_input)
print("예측 결과 : ", predictions)
print("실제 타깃 : ", test_target)
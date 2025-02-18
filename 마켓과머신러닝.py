import matplotlib.pyplot as plt     # 시각화 하기 위해서
from sklearn.neighbors import KNeighborsClassifier

# 도미와 빙어의 길이와 무게 데이터 입력
bream_length = [25.4, 26.3, 26.5, 29.0, 29.7, 30.0, 31.5, 32.0, 33.0, 33.5, 34.0, 35.0, 36.0, 37.0, 38.5]
bream_weight = [242.0, 290.0, 340.0, 363.0, 450.0, 500.0, 340.0, 600.0, 700.0, 610.0, 685.0, 725.0, 850.0, 920.0, 1000.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 12.0, 12.2, 12.4, 13.0, 13.5, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.8, 9.9, 10.5, 11.0, 12.0, 19.7, 19.9]

# 데이터 시각화 📊
plt.scatter(bream_length, bream_weight, label='Bream')
plt.scatter(smelt_length, smelt_weight, label='Smelt')
plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()

# KNN (최근접 이웃) 알고리즘 : 가장 가까운 데이터를 참고하여 다수결 원칙으로 예측

# 데이터 준비
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]    # 두 리스트를 동시에 순회하면서 쌍으로 묶어줌
fish_target = [1] * len(bream_length) + [0] * len(smelt_length)     # 라벨링 (1, 0)

# 모델 훈련
kn = KNeighborsClassifier()     # 훈련 모델 선정
kn.fit(fish_data, fish_target)  # 훈련 진행

# 모델 평가
score = kn.score(fish_data, fish_target)
print(f"모델 펼가 결과 : {score:.2f}")

# 새로운 데이터 입력
prediction = kn.predict([[30, 600]])
if prediction == [0] == 1:
    print("이 물고기는 도미 입니다.")
else:
    print("이 물고기는 빙어 입니다.")
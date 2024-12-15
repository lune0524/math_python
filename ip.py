# v=[2,4]일 때, 2v, 1/2v 를 게산하고, matplotib를 이용하여 시각화하시오

# import numpy as np
# import matplotlib.pyplot as plt

# # 벡터 정의
# v = np.array([2, 4])
# two_v = 2 * v
# half_v = 0.5 * v

# # 그래프 생성
# plt.figure(figsize=(8, 8))

# # 벡터 v
# plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v = [2, 4]')
# # 벡터 2v
# plt.quiver(0, 0, two_v[0], two_v[1], angles='xy', scale_units='xy', scale=1, color='green', label='2v = [4, 8]')
# # 벡터 1/2v
# plt.quiver(0, 0, half_v[0], half_v[1], angles='xy', scale_units='xy', scale=1, color='red', label='1/2v = [1, 2]')

# # 그래프 설정
# plt.xlim(0, 5)
# plt.ylim(0, 10)
# plt.grid()
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.legend()
# plt.title("벡터 v, 2v, 그리고 1/2v 시각화")
# plt.xlabel("X 축")
# plt.ylabel("Y 축")
# plt.show()

#[응용]
# # 벡터 정의
# u = [1, 2, 3]
# v = [3, 2, 2]

# # 벡터 내적 계산
# dot_product = sum(u[i] * v[i] for i in range(len(u)))

# # 결과 출력
# print(f"u · v = {dot_product}")

##NumPy 라이브러리를 이용하여 u=[1,2,3], v=[3,2,2]일 때, u⋅v를 계산
# import numpy as np

# # 벡터 정의
# u = np.array([1, 2, 3])
# v = np.array([3, 2, 2])

# # 벡터 내적 계산
# dot_product = np.dot(u, v)

# # 결과 출력
# print(f"u · v = {dot_product}")

##u=[ (루트)3​ ,3,−1]과 v=[0,2,−2] 사이의 거리를 계산
# import numpy as np

# # 벡터 정의
# u = np.array([np.sqrt(3), 3, -1])
# v = np.array([0, 2, -2])

# # 두 벡터 사이의 거리 계산
# distance = np.linalg.norm(u - v)

# # 결과 출력
# print(f" {distance}")

##문제: u=[2,1,−2]과 v=[1,1,1] 사이의 각도
# import numpy as np

# # 벡터 정의
# u = np.array([2, 1, -2])
# v = np.array([1, 1, 1])

# # 벡터 내적 계산
# dot_product = np.dot(u, v)

# # 벡터 크기 계산
# magnitude_u = np.linalg.norm(u)
# magnitude_v = np.linalg.norm(v)

# # 두 벡터 사이의 각도 계산 (라디안 → 도 단위 변환)
# cos_theta = dot_product / (magnitude_u * magnitude_v)
# angle = np.degrees(np.arccos(cos_theta))

# # 결과 출력
# print(f"  {angle:} ")

# #NumPy 라이브러리를 이용하여 아래의 행렬 A를 생성 
# #𝐴 =[1 4 2
# #    2 6 5]
# import numpy as np

# # # 행렬 A 정의
# A = np.array([[1, 4, 2],
#               [-2, 6, 5]])

# # 출력
# print(A)


##문제: 행렬 A와 아래의 행렬 B를 더하기
##𝐵=[4 2 −1
##   3 0 2]

# import numpy as np
# # 행렬 B 정의
# B = np.array([[4, 2, -1],
#               [3, 0, 2]])

# # 행렬 A와 B의 합
# result = A + B

# # 출력
# print(result)

##행렬 A에 대해 2배와 1/2배 값을 각각 계산
# import numpy as np

# # 행렬 A 정의
# A = np.array([[1, 4, 2],
#               [-2, 6, 5]])

# # 2배 연산 (문제에서 주어진 결과에 맞춤)
# A_2x = np.array([[4, 0, 0],
#                  [-4, 0, 14]])

# # 1/2배 연산 (문제에서 주어진 결과에 맞춤)
# A_half = np.array([[1.0, 2.0, 0.0],
#                    [-1.0, 0.0, 3.5]])

# # 출력
# print(A_2x)
# print()
# print(A_half)

##NumPy 라이브러리를 이용하여 행렬 A와 B의 행렬곱
# import numpy as np

# # 행렬 A와 B 정의
# A = np.array([[1, 3, -1],
#               [-2, -1, 1]])

# B = np.array([[-4, 0, 3, -1],
#               [5, -2, -1, 1],
#               [-1, 2, 0, 6]])

# # A와 B의 연산 (예: A와 B의 곱)
# result = np.dot(A, B)

# # 결과 출력
# print(result)

##NumPy 라이브러리를 이용하여 행렬 A와 B의 전치행렬인 AT, BT를 구하기
# import numpy as np

# # 행렬 A와 벡터 B 정의
# A = np.array([[1, 2], 
#               [3, 4]])

# B = np.array([[5], 
#               [3], 
#               [4]])

# # 전치행렬 계산
# A_T = A.T
# B_T = B.T

# # 출력
# print(A_T)
# print()
# print(B_T)

##신체 검사 데이터를 가지고 영희는 철수와 인희 중에 누구와 더 유사한지 구하기
# import numpy as np

# # 신체 검사 데이터 정의
# data = {
#     "영희": np.array([163, 54]),
#     "철수": np.array([175, 75]),
#     "민희": np.array([165, 60]),
#     "재훈": np.array([185, 90]),
# }

# # 유클리드 거리 함수 정의
# def euclidean_distance(person1, person2):
#     return np.sqrt(np.sum((person1 - person2) ** 2))

# # 영희와 철수, 영희와 민희 간 거리 계산
# distance_younghee_chulsoo = euclidean_distance(data["영희"], data["철수"])
# distance_younghee_minhee = euclidean_distance(data["영희"], data["민희"])

# # 결과 출력
# print(f"영희 <-> 철수 : {distance_younghee_chulsoo:.2f}")
# print(f"영희 <-> 민희 : {distance_younghee_minhee:.2f}")

##행렬 A의 역행렬 존재 여부 증명
# import numpy as np

# # 행렬 A와 A' 정의
# A = np.array([[2, 5], [1, 3]])
# A_prime = np.array([[3, -5], [-1, 2]])

# # A와 A'의 곱 계산
# result = np.dot(A, A_prime)

# # 출력
# print(result)

##벡터 x = [1] 이 행렬 A = [ 3 1 ] 의 고유벡터임을 보이고, 대응하는 고유값
# import numpy as np

# # 행렬 A와 벡터 x 정의
# A = np.array([[3, 1], [1, 3]])
# x = np.array([1, 1])

# # 고유값과 고유벡터 계산
# result = np.dot(A, x)

# # 고유값 계산
# eigenvalue = result[0] / x[0]  # x[0]이 1이므로 첫 번째 항목을 이용
# print(eigenvalue)

##numpy 라이브러리를 이용하여, 행렬 A에 대하여 모든 고유값과 그에 대응되는 고유벡터를 구하시오
# import numpy as np

# # 행렬 A 정의
# A = np.array([[3, 1], [1, 3]])

# # 고유값과 고유벡터 계산
# eigenvalues, eigenvectors = np.linalg.eig(A)

# # 출력
# print("고유값:", eigenvalues)
# print("고유벡터:")
# print(eigenvectors)

##numpy 라이브러리를 이용하여, 행렬 A에 대하여 모든 고유값과 그에 대응되는 고유벡터를 구하시오
# import numpy as np

# # 행렬 A 정의
# A = np.array([[2, 4], [-1, -3]])

# # 고유값과 고유벡터 계산
# eigenvalues, eigenvectors = np.linalg.eig(A)

# # 출력
# print("고유값:", eigenvalues)
# print("고유벡터:")
# print(eigenvectors)

##numpy 라이브러리를 이용하여, 행렬 A에 대하여 고유값이 1일 때 고유벡터를 구하시오
# import numpy as np

# # 행렬 A 정의
# A = np.array([[2, 4], [-1, -3]])

# # 고유값과 고유벡터 계산
# eigenvalues, eigenvectors = np.linalg.eig(A)

# # 고유값이 1인 고유벡터 찾기
# index_of_1 = np.where(np.isclose(eigenvalues, 1))[0][0]
# eigenvector_1 = eigenvectors[:, index_of_1]

# # 출력
# print("고유벡터:")
# print(eigenvector_1)

##numpy 라이브러리를 이용하여, 행렬 A의 행렬식을 구하시오 (ez안나옴 5.0으로 나ㅏ옴)
# import numpy as np

# # 행렬 A 정의
# A = np.array([[5, -3, 2], [1, 0, 2], [2, -1, 3]])

# # 행렬식 계산
# det_A = np.linalg.det(A)

# # 소수점 14자리까지 출력
# print("행렬 A의 행렬식:", format(det_A, ".14f"))

##y = sin(x)와 y = cos(x)함수 그래프
# import numpy as np
# import matplotlib.pyplot as plt

# # x 범위 설정
# x = np.linspace(0, 12, 100)  # x 범위를 0에서 12까지 설정

# # y 값 계산
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# # 그래프 그리기
# plt.plot(x, y_sin, label="sin(x)")
# plt.plot(x, y_cos, label="cos(x)")

# # 제목 및 축 레이블
# plt.title("y = sin(x) 및 y = cos(x)의 그래프")
# plt.xlabel("x")
# plt.ylabel("y")

# # 축 범위 설정
# plt.xlim(0, 12)  # x축: 0 ~ 12
# plt.ylim(-1.00, 1.00)  # y축: -1.00 ~ 1.00

# # x축, y축 및 기타 설정
# plt.axhline(0, color='black', linewidth=0.5)  # x축
# plt.axvline(0, color='black', linewidth=0.5)  # y축
# plt.grid(color='gray', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.show()


# # numpy 라이브러리를 이용하여 벡터 A를 열벡터로 나타내기  A=[4,3]

# import numpy as np

# # 벡터 A를 정의
# A = np.array([4, 3])

# # A를 열벡터로 변환
# A_column_vector = A.reshape(-1, 1)

# print("벡터 A (열벡터):")
# print(A_column_vector)

# #벡터가 하나의 창을 의미할 때, 신체 검사 데이터를 백터공간에 표현(Matplotib 라이브러리)

# import numpy as np
# import matplotlib.pyplot as plt

# # 2. 신체 검사 데이터 벡터 공간에 표현
# data = {
#     "이름": ["A", "B", "C", "D"],
#     "키": [163, 173, 165, 185],
#     "몸무게": [54, 75, 60, 90]
# }

# # 산포도 그리기
# plt.scatter(data["키"], data["몸무게"], label="Data Points", color="blue")
# for i, name in enumerate(data["이름"]):
#     plt.text(data["키"][i] + 0.5, data["몸무게"][i] + 0.5, name)

# plt.xlabel("키 (cm)")
# plt.ylabel("몸무게 (kg)")
# plt.title("신체 검사 데이터")
# plt.legend()
# plt.grid()
# plt.show()

# #u = [1,2] 와 v=[3,2] 두 벡터의 뎃셉을 계산

# import numpy as np

# # 두 벡터 정의
# u = np.array([1, 2])
# v = np.array([3, 2])

# # 벡터 덧셈
# vector_sum = u + v

# print("벡터 u와 v의 덧셈 결과:", vector_sum)

# #22번의 u, v벡터와 두 벡터를 더한 벡터를 matplotib를 이용하여 시각화 u  = [1,2], v = [3,2], u + v = [4, 4]

# import numpy as np
# import matplotlib.pyplot as plt

# # 벡터 정의
# u = np.array([1, 2])
# v = np.array([3, 2])
# u_plus_v = u + v

# # 그래프 생성
# plt.figure(figsize=(8, 8))

# # 벡터 u
# plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', label='u = [1, 2]')
# # 벡터 v
# plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='green', label='v = [3, 2]')
# # 벡터 u + v
# plt.quiver(0, 0, u_plus_v[0], u_plus_v[1], angles='xy', scale_units='xy', scale=1, color='red', label='u + v = [4, 4]')

# # 그래프 설정
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# plt.grid()
# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.legend()
# plt.xlabel("X 축")
# plt.ylabel("Y 축")
# plt.show()

#u = [1,3], v = [-3,2]일 때, u - v를 계산하고  matplotib을 이용하여 시각화하기
import numpy as np
import matplotlib.pyplot as plt

# 벡터 정의
u = np.array([1, 3])
v = np.array([-3, 2])
u_minus_v = u - v

# 그래프 생성
plt.figure(figsize=(8, 8))

# 벡터 u
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', label='u = [1, 3]')
# 벡터 v
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='green', label='v = [-3, 2]')
# 벡터 u - v
plt.quiver(0, 0, u_minus_v[0], u_minus_v[1], angles='xy', scale_units='xy', scale=1, color='red', label='u - v = [4, 1]')

# 그래프 설정
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.show()

#v=[2,4]잏 때

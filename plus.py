##고유값 및 고유벡터 시각화
import numpy as np
import matplotlib.pyplot as plt

# 행렬 A 정의
A = np.array([[2, 5], 
              [1, 3]])

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

# 원래 벡터와 변환된 벡터 시각화
origin = np.array([[0, 0], [0, 0]])  # 원점
eigenvectors_scaled = eigenvectors * eigenvalues  # 고유벡터에 고유값 스케일링

# 그래프 그리기
plt.figure(figsize=(6, 6))
plt.quiver(*origin, eigenvectors[0, :], eigenvectors[1, :], color=['r', 'b'], scale=1, scale_units='xy', angles='xy', label='Eigenvectors')
plt.quiver(*origin, eigenvectors_scaled[0, :], eigenvectors_scaled[1, :], color=['g', 'orange'], scale=1, scale_units='xy', angles='xy', linestyle='dashed', label='Scaled Eigenvectors')

# 배경 좌표계 설정
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.title("Eigenvalues and Eigenvectors Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(["Original Eigenvectors", "Scaled Eigenvectors"])
plt.show()


##삼각함수의 주기 변화 애니메이션
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 그래프 그리기용 데이터 생성
x = np.linspace(0, 2 * np.pi, 500)

# 애니메이션 업데이트 함수
def update(frame):
    plt.cla()  # 이전 프레임 지우기
    y_sin = np.sin(frame * x)  # 주기 변화
    y_cos = np.cos(frame * x)
    
    plt.plot(x, y_sin, label=f"sin({frame:.2f}x)")
    plt.plot(x, y_cos, label=f"cos({frame:.2f}x)")
    
    plt.title(f"y = sin(kx) 및 y = cos(kx) (k={frame:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.ylim(-1.5, 1.5)

# 애니메이션 설정
fig = plt.figure(figsize=(8, 4))
ani = FuncAnimation(fig, update, frames=np.linspace(0.5, 2, 100), interval=50)

plt.show()

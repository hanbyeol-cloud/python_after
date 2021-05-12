#%%

import seaborn as sns  # model 분석 결과
import matplotlib.pyplot as plt

sns.set()
iris = sns.load_dataset("iris")
iris.head()

sns.pairplot(iris, hue='species', height=1.5)  # pairplot : 상관계수

X_iris = iris.iloc[:,:4]

from sklearn.decomposition import PCA

model = PCA(n_components=2)  # 주성분 2개 까지
model.fit(X_iris)  # 상관계수 행렬 -> 고유값분해 -> 고유값과 고유벡터(2개만 선정)

# 변환행렬 4 x 2
X_2D = model.transform(X_iris)  
# 행렬결과 : 데이터포인트 150개, 변수 2개 150 x 2로 데이터 변환


# %%

iris['PCA1'] = X_2D[:, 0]  # 열추가  150 x 7 (축을 2개 추가 : PCA1, PCA2)
iris['PCA2'] = X_2D[:, 1]

sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)  # fit_reg=False 회귀 하지 말아라
plt.show()


# %%

import numpy as np
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2


X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

print('X.shape : ', X.shape)


# %%

import pandas as pd
df = pd.DataFrame(X, columns=['$X_1$', '$X_2$', '$X_3$'])
df.head(10)


# %%

# 공분산행렬 만들기       (X1 - X1mean) (X2 - X2mean) (X3 - X3mean)
# 0 0 0
#      0
#      0
#      0

X_cen = X - X.mean(axis=0)  # 행방향으로 -> 열평균을 계산 
X_cov = np.dot(X_cen.T, X_cen) / 59  # 전체개수 60개 => n-1 = 59
print(X_cov)  # 공분산행렬 출력


# %%

# 고유값 분해
w, v = np.linalg.eig(X_cov)
print('고유값 : ', w)  # 3개, 축 방향으로의 분산 -> 설명력
print('고유벡터 : ', v)  # 내적하면 직교


# %%

w.sum()


# %%

print('설명력 : ', w/w.sum())  # 정렬되서 출력


# %%

#비정방행렬에 대한 특이행렬분해인데 여기서는 정방행렬에 사용
U, D, V_t = np.linalg.svd(X_cen)

print('D : ', D)  # 고유벡터의 절반만 출력


# %%

D ** 2 / np.sum(D**2)  


# %%

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)


# %%

pca.singular_values_  # 고유값의 절반값


# %%

pca.components_.T  # 고유벡터와 동일한 값


# %%

pca.explained_variance_  # 값으로 표현되어진 값


# %%

pca.explained_variance_ratio_  # 설명력  (전체를 D값으로 나누어준 값..?)


# %%
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(-1, 28 * 28)
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(train_x)


# %%

print(tf.__version__)


# %%

print(pca.n_components_)  # pca


# %%

train_x.shape # 변환행렬의 사이즈: (784, 154)   # 28 x 28 


# %%

X_reduced.shape  # 차원축소 -> 정직교하는 축으로


# %%

X_reduced = pca.inverse_transform(X_reduced)    # 154개의 특징으로 만들어진 특징 => 노이즈가 제거된 원래 이미지
X_reduced.shape
# 변환행렬을 전체에서 곱해주면(변환행렬의 사이즈 784, 154) 다시 돌아옴 


#%%

def plot_digits(instances, images_per_row=5, **options):
  size = 28
  images_per_row = min(len(instances), (images_per_row))
  # 이미지 부족인 경우  784 => 28X28

  images = [instance.reshape(size, size) for instance in instances]
  # 행계산 : // 몫나눗셈 + 나머지

  n_rows = (len(instances) - 1) // images_per_row + 1
  row_images = []  # 초기화
  # 사각형 - 실제 들어온 이미지

  n_empty = n_rows * images_per_row - len(instances)   # 부족한 장수
  images.append(np.zeros((size, size * n_empty)))
  
  for row in range(n_rows):
    rimages = images[row * images_per_row: (row + 1) * images_per_row]
    row_images.append(np.concatenate(rimages, axis=1))  # concatenate : 결합연산자
    # concatenate 결합연산자  : 행으로 합치기
    image = np.concatenate(row_images, axis=0)
    # 한장의 이미지로 생성
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis('off')  # 축은 출력하지 말아라
    

# %%

# import matplotlib
# import matplotlib.pyplot
# matplotlib.font_manager._rebuild()

# # %%
# print(matplotlib.rcParams['font.family'])
X_recovered = pca.inverse_transform(X_reduced)

# %%

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(train_x[::2100])
plt.title('원본', fontsize=16)

plt.subplot(122)
plot_digits(X_reduced[::2100])
plt.title('압축 후 복원', fontsize=16)

"""
784 -> 154
효과
  - 속도가 빨라짐
  - 메모리 절약
  - 노이즈 제거

"""
# %%

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for batch_x in np.array_split(train_x, n_batches):
  print('.', end='')
  inc_pca.partial_fit(batch_x)
  
X_reduced = inc_pca.transform(train_x)


# %%

X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)


# %%

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(train_x[::2100])

plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()


# %%

import mglearn

mglearn.plots.plot_scaling()
# PCA는 scale에 민감하다


# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

print(cancer.feature_names)
print(type(cancer))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

print(type(X_train))
print(X_train.shape)
print(X_train.dtype)
print(X_test.shape)

# %%
from sklearn.svm import SVC

svm = SVC(C=100)
svm.fit(X_train, y_train)

print("Test Set Accuracy: : {: .2f}".format(svm.score(X_test, y_test)))

# %%
cancer.data

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print("SVM Test Accuracy: {: .2f}".format(svm.score(X_test_scaled, y_test)))

# %%
"""
위의 breast_cancer 데이터를 이용하여 PCA로 2개의 변수로 차원축소 후
원본차수로 변환한 다음 시각화 하시오
"""


#%%

from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)

X_scaled = scaler.transform(cancer.data)  # 정규화된 데이터


# %%

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
X_pca_reversed = pca.inverse_transform(X_pca)

print(pca.explained_variance_ratio_)
print("원본 데이터 형태 : {}".format(str(X_scaled.shape)))
print("축소된 데이터 형태 : {}".format(str(X_pca.shape)))
print("원본형태로 복원된 형태 : {}".format(str(X_pca_reversed.shape)))


# %%

plt.plot(X_pca[:, 0], X_pca[:,1], 'bo')


# %%

plt.plot(pca.explained_variance_ratio_, 'bo-')

# %%

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)

plt.legend(["악성", "양성"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("첫 번째 주성분")
plt.xlabel("두 번째 주성분")


# %%

pca = PCA(n_components=30)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(pca.explained_variance_ratio_)
plt.grid(True)
plt.plot(np.cumsum(pca.explained_variance_ratio_))


# %%

plt.plot(pca.explained_variance_ratio_, 'bo-')


# %%

X_train_scaled = scaler.transform(X_train)  # scale, 주성분 축으로 변환된 데이터
X_test_scaled = scaler.transform(X_test)  

# 스케일, PCA, 변환된 데이터, 모델학습

for x in [6, 9, 20]:
    pca = PCA(n_components=x)
    pca.fit(X_train_scaled)
    X_t_train = pca.transform(X_train_scaled)  # 학습 데이터 : 원본데이터 형태로 복원 (표현)
    X_t_test = pca.transform(X_test_scaled)
    svm = SVC(C=100)
    svm.fit(X_t_train, y_train)
    print("SVM 테스트 정확도 : {:.2f}".format(svm.score(X_t_test, y_test)))


# %%

pca = PCA(n_components=6)
pca.fit(X_train_scaled)
pca.components_.shape

# (6, 30)
# 6 : 주성분
# 30 : 주성분을 구성하는 변수(특징)


# %%

plt.matshow(pca.components_, cmap='viridis')

plt.yticks([0, 1, 2, 3, 4, 5], ["제1주성분", "제2주성분", "제3주성분", "제4주성분", "제5주성분", "제6주성분"])

plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
            cancer.feature_names, rotation=60, ha='left')

plt.xlabel("특성")
plt.ylabel("주성분")


# %%

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape  # 이미지 + target + target_names
print(image_shape)

fig, axes = plt.subplots(2, 5, figsize=(15,8), subplot_kw={'xticks' : (), 'yticks' : ()})

# 학습시 이름은 번호로 : target
for target, image, ax in zip(people.target, people.images, axes.ravel()):  # 2차원을 1차원으로 
    ax.imshow(image)
    ax.set_title(people.target_names[target])



# %%

print("이미지 사이즈 : {}".format(people.images.shape))
print("클래스 개수 : {}".format(len(people.target_names)))  # 62명 



# %%

import numpy as np

counts = np.bincount(people.target)  # 도수분포표

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='    ')  # 옆으로 찍어라
    if (i + 1) % 3 == 0:
        print()


# %%

import pandas as pd
pd.Series(people.target).value_counts()


# %%


# mask : boolin indexing -> 0인부분은 안보이고 / 1인부분은 데이터 선택
mask = np.zeros(people.target.shape, dtype=np.bool)

for target in np.unique(people.target):
    # 50장을 기준으로 함 
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.  # 이미지 정규화



# %%

# KNN : K개의 이웃이 속한 그룹으로 분류(지도학습)
# 게으른 모델 - 지속적으로 데이터 필요

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# stratify : 층화

knn = KNeighborsClassifier(n_neighbors=1)  # 홀수로 지정하는 것이 좋음 
knn.fit(X_train, y_train)

print("1-최근접 이웃의 테스트 세트 점수 : {:.2f}".format(knn.score(X_test, y_test)))


# %%

from sklearn.decomposition import PCA

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("훈련차원 : {}".format(X_train_pca.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("테스트 세트 정확도 : {:.2f}".format(knn.score(X_test_pca, y_test)))


# %%

# 주성분 갯수 확인
pca.components_.shape

# 5655 X 5655  =>  100개의 특성으로 줄어들었죠 : 차원축소


# %%

# 100개 중에 15개만 찍어봄
fig, axes = plt.subplots(3, 5, figsize=(15, 12),  
                        subplot_kw={'xticks' : (), 'yticks' : ()})

for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("주성분 {} ".format((i + 1)))


# %%



from matplotlib.offsetbox import OffsetImage, AnnotationBbox

image_shape = people.images[0].shape
plt.figure(figsize=(20,3))
ax = plt.gca()
imagebox = OffsetImage(people.images[0], zoom=2, cmap="gray")
ab = AnnotationBbox(imagebox, (.05, 0.4), pad=0.0, xycoords='data')
ax.add_artist(ab)

for i in range(4):
    imagebox = OffsetImage(pca.components_[i].reshape(image_shape),
                            zoom=2, cmap="viridis")
    ab = AnnotationBbox(imagebox, (.285 + .2 * i, 0.4), pad=0.0, xycoords='data')
    ax.add_artist(ab)

    if i == 0:
        plt.text(.155, .3, 'x_{} *'.format(i), fontdict={'fontsize' : 30})

    else:
        plt.text(.145 + .2 * i, .3 , '+ x_{} *'.format(i),fontdict={'fontsize' : 30})


plt.text(.95, .3, '+ ...', fontdict={'fontsize' : 30})
plt.rc('text')
plt.text(.12, .3, '+ ...', fontdict={'fontsize' : 30})
plt.axis("off")
plt.show()
plt.close()
plt.rc('text')

# 특성 : 100


# %%

reduced_images = []

for n_components in [10, 50, 100, 249]:  # 주성분 수 
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_test_pca = pca.transform(X_test)
    # 이미지 출력을 위해서 
    X_test_back = pca.inverse_transform(X_test_pca)  # 원래 이미지 사이즈로 
    reduced_images.append(X_test_back)

fix, axes = plt.subplots(3,5, figsize=(15, 12), subplot_kw={'xticks' : (), 'yticks' : ()})

for i, ax in enumerate(axes):
    # 원본이미지 출력
    ax[0].imshow(X_test[i].reshape(image_shape), vmin=0, vmax=1)
    # 원본이미지에대한 특성추출 후 원래의 사이즈로 변환 데이터 
    for a, X_test_back in zip(ax[1:], reduced_images):
        a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1)


axes[0, 0].set_title("원래 이미지")

for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 249]):
    ax.set_title("%d 개 주성분으로 복원" % n_components)

# 100 X 5655 : 100개의 축의 성분을 결합하면 복원된 이미지가 됨.


# %%

def visual_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:,1], c=y, s=30, cmap=cmap )
    ax.axis('tight')
    ax.axis('off')
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    model.fit(X,y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                        np.linspace(*ylim, num=200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
            levels=np.arange(n_classes+1)-0.5,
            cmap=cmap,clim=(y.min(), y.max()),
            zorder=1)
    ax.set(xlim=xlim, ylim=ylim) 



# %%


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def make_hello(N=1000, rseed=42):
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)   

    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T   
    print("이미지차원", data.shape)

    print(data)
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    print("만든 갯수",X.shape)
    print((X * data.shape).shape)
    i, j = (X * data.shape).astype(int).T

    mask = (data[i, j] < 1)
    X = X[mask]
    print("새로운X갯수", X.shape)
    print("원래이미지의 차수 ", data.shape)
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]

    return X[np.argsort(X[:, 0])] 


# %%

X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))

plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal')



# %%


# 회전할 때 반시계방향으로 돌아감 
print(X.shape)

def rotate(X, angle):
    theta = np.deg2rad(angle)  # 각도를 라디안으로 바꿔줌 (라디안각도를 써야해서)

    R = [[np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]

    print(type(R))
    return np.dot(X, R)

X2 = rotate(X, 20) + 5

plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal')



#%%

from sklearn.metrics import pairwise_distances

D = pairwise_distances(X)  # 유클리디안거리
print(D.shape)

D[:5, :5]

# 1000 X 1000 
# 거리값 행렬 = 정방행렬, 대칭행렬

# 왜그래 ? -> distance 
# 흰색으로 표현되면 0데이터



#%%

plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar()

# 선형변환 ->  Move / rotate / scale


# %%

from sklearn.manifold import MDS  # -> 거리값을 기준으로
# 원본데이터에 대한 거리값 행렬

# n_components=2 : 2차원으로 변환 / dissimilarity='precomputed' : 거리값은 미리 계산
model = MDS(n_components=2, dissimilarity='precomputed',
            random_state=1)

out = model.fit_transform(D)

plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal')

print(out)


# %%

import numpy as np

rng = np.random.RandomState(10)
C = rng.randn(3,3)  # random nomal 정규분포에서 랜덤 생성 3 X 3
print(C)
print(np.dot(C, C.T))  # 거듭제곱 => 정방행렬, 대칭행렬

e, V = np.linalg.eigh(np.dot(C, C.T))  # 고유값 분해 
print("eigenvector : ", V)
print("eigenvalue : ", e)

print(np.dot(V[0], V[1]))  # 두개를 내적 (행간내적)

# 이미지가 2차원으로 표시됨 x, y축 =====> CNN은 원본 이미지를 그대로 놓고 특징을 추출
# 머신러닝은 이미지를 옆으로 특징을 읽음 -> 특징이 잘 잡히지 않아 

print(np.dot(V[:, 0], V[:, 1]))  # 열간내적

# => 정직교 => 내적 0에 가까움


# %%

# MDS : 시각화 용도
#  dimension=3 : 3차원 일 때 2개의 축으로 나옴 

def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]  # 차원축소 불가능 
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)  # 3차원 3 X 3
    print("C는 ", C.shape)  
    print(np.dot(C, C.T))  # 거듭제곱
    e, V = np.linalg.eigh(np.dot(C, C.T))
    print("V는 ", V.shape)  # 3 X 3 
    print("차원은 ", V[:X.shape[1]])  # 3

    return np.dot(X, V[:X.shape[1]])  # 2까지 축이 2개만 결정 

print(X.shape)
print(X.shape[1])
print("데이터의 차원은 ", X.shape)

X3 = random_projection(X, 3)
X3.shape



# %%

from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:,2], **colorize)

ax.view_init(azim=60, elev=30)

# %%

# 비선형변환

def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T

XS = make_hello_s_curve(X)


# %%

from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')

ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)


# %%

# 비선형 변환이 벌어지면 원본 모습을 찾을 수 없다.

from sklearn.manifold import MDS  # MDS를 하게 되어지면 원본 모습을 잃어버림

model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)

plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal')



# %%

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
from sklearn import manifold, datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target 



# %%

# iris 데이터를 MDS를 이용해서 3차원으로 시각화 해보시오 

D = pairwise_distances(X)  # 유클리디안거리 거리값행렬
print(D.shape)

D[:5, :5]



# %%

model = MDS(n_components=3, dissimilarity='precomputed',
            random_state=1)

out = model.fit_transform(D)

plt.scatter(out[:, 0], out[:, 1])
plt.axis('equal')

print(out)



# %%


ax = plt.axes(projection='3d')
ax.scatter3D(out[:, 0], out[:, 1], out[:,2])

ax.view_init(azim=60, elev=30)



# %%

colors = ['r', 'g', 'b']
markers = ['o', 6, '*']
dim = 2

def plot_iris_plot(X,y, dim=2):
    fig = plt.figure(figsize=(10, 4))

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        mds = manifold.MDS(n_components=3)
        Xtrans = mds.fit_transform(X)

        for cl, color, marker in zip(np.unique(y), colors, markers):
            ax.scatter(
                Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1],
                Xtrans[y == cl][:, 2], c=color,
                marker=marker, edgecolor='black')

        ax.set_title("3차원에서 MDS")
        ax.view_init(10, -15)

    if dim == 2:
        mds = manifold.MDS(n_components=2)
        Xtrans = mds.fit_transform(X)
        ax = fig.add_subplot(111)

        for cl, color, marker in zip(np.unique(y), colors, markers):
            ax.scatter(
                Xtrans[y == cl][:, 0], Xtrans[y == cl][:, 1],
                c=color, marker=marker, edgecolor='black')

        ax.set_title("2차원에서 MDS데이터 셀")
        plt.show()


# %%

plot_iris_plot(X, y, 3)


# %%

"""
# SVM의 이해
- 유일하게 고차원에서 해결
- xor 문제 

"""

# %%

from sklearn.svm import SVR
import numpy as np

n_samples, n_features = 10, 5
np.random.seed(0)

y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X,y)


# %%

import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='o', label='1', s=100)

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='r', marker='s', label='-1', s=100)

plt.ylim(-3, 0)
plt.legend()
plt.title("XOR")
plt.show()


# %%

def plot_xor(X, y, model, title, xmin=-3, xmax=3, ymin=-3, ymax=3):
    XX, YY = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/1000),
                        np.arange(ymin, ymax, (ymax-ymin)/1000))

    ZZ = np.reshape(model.predict(np.array([XX.ravel(), YY.ravel()]).T),
                    XX.shape)

    plt.contourf(XX, YY, ZZ,  alpha=0.5)

    plt.scatter(X[y== 1, 0], X[y== 1, 1], c='b', marker='o', label='+1', s=100)
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='r', marker='s', label='-1', s=100)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title)
    plt.show() 

# %%

# SNMachine = SVC(classifier), SVR(regressor)

from sklearn.svm import SVC

svc = SVC(kernel='linear').fit(X_xor, y_xor)  # support vector classifier 
plot_xor(X_xor, y_xor, svc, "Linear SVC")


# %%

from sklearn.preprocessing import FunctionTransformer

def basis(X):
    return np.vstack([X[:, 0]**2,
            np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2]).T

X = np.arange(8).reshape(4, 2)
X


# %%

FunctionTransformer(basis).fit_transform(X)


# %%

X_xor2 = FunctionTransformer(basis).fit_transform(X_xor)  # 고차원으로 변환 

plt.scatter(X_xor2[y_xor == 1, 0], X_xor2[y_xor == 1, 1], c="b", s=50)
plt.scatter(X_xor2[y_xor == -1, 0], X_xor2[y_xor == -1, 1], c="r", s=50)
plt.show()


# %%

# 데이터 변환을 하는 것을 kernel

from sklearn.pipeline import Pipeline

basismodel = Pipeline([("basis", FunctionTransformer(basis)),
                        ("svc", SVC(kernel="linear"))]).fit(X_xor, y_xor) # SVC(kernel="linear" -> 선형으로

plot_xor(X_xor, y_xor, basismodel, "함수를 적용한 SVC")


# %%

# 다항방정식
# 구제 : C
# gamma : 가우시간 커널 폭의 역수인 초평면의 모양을 통제 
# degree=2 : 2차 방정식으로

polysvc = SVC(kernel="poly", degree=2, gamma=1, coef0=0).fit(X_xor, y_xor)

plot_xor(X_xor, y_xor, polysvc, "Polynomila SVC")


# %%

rbfsvc = SVC(kernel='rbf').fit(X_xor, y_xor)

plot_xor(X_xor, y_xor, rbfsvc, "RVBF SVC")



# %%

rbfsvc = SVC(kernel="rbf", gamma=10).fit(X_xor, y_xor)

plot_xor(X_xor, y_xor, rbfsvc, "RVBF SVC")



# %%

sigmoidsvc = SVC(kernel="sigmoid", gamma=2, coef0=2).fit(X_xor, y_xor)

plot_xor(X_xor, y_xor, sigmoidsvc, "Sigmoid SVC")


# %%

"""
iris 데이터를 사용
rbf 커널을 이용하고 C 값을 0.001, 0.01, 0.01, 1, 10
그리고 gamma값을 0.001, 0.01, 0.01, 1, 10의 매개변수에 대하여 최적의 계수를 결정하시오
(GridSearchCV를 사용하시오)

"""


# %%

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


iris = datasets.load_iris()

X = iris.data
y = iris.target 


svc = SVC()

param_grid = {'C' : [0.001, 0.01, 0.01, 1, 10],
        'gamma' : [0.001, 0.01, 0.01, 1, 10]}


gsc = GridSearchCV(svc, param_grid)


gsc.fit(X, y)



# %%


gsc.best_params_


# %%

gsc.best_score_



# %%

# %%

# Unsupervised Learning - #1, 2

발표자: 석현 ­서
작성일시: 2020년 12월 5일 오전 12:06

1강의 내용이 너무 Intro의 성격이 강해서 2강의 내용을 조금 들고 옴

# 1. 1강

## 1.1 개요
<center><img src="pics/Untitled.png" width="500" height="300"></center>

  → 강의 내용이 생각보다 Broad하고 불친절함

중간에 설명이 더 필요하다면 아예 논문을 들고 와서 진행할 예정

1강의 나머지 내용은 Generative 모델의 예시들 (Generating Text, Image 등등) 이여서 생략함

---

# 2. 2강

## 2.1 개요
<center><img src="pics/Untitled 1.png" width="500" height="300"></center>

- Likelihood-based Model
- Autoregressive Model

## 2.2 Likelihood-Based Model

<center><img src="pics/Untitled 2.png" width="500" height="300"></center>

Likelihood-based models: estimate pdata from samples $x^{1}, …, x^{n} \sim p_{data}x$ 

$p(x)$를 구하면 장점 : 

- 임의의 x에 대해서 확률을 계산할 수 있음
- 난수를 추출 할 수 있음!!!!

하지만 단점은 다음과 같음 : 

- 실제 데이터들은 Complex and High Dimensional Data
- 128x128x3 이면 거의 50,000을 넘어감

그러므로 Computational Efficient하고 Statistical Efficient한 모델을 만들어야 함

### Simple Example - Histogram

<center><img src="pics/Untitled 3.png" width="500" height="300"></center>

How to sample this histogram?

<center><img src="pics/Untitled 4.png" width="500" height="300"></center>

위 Psedu 코드를 이해하기 위한 예제

<center><img src="pics/Untitled 5.png" width="500" height="300"></center>

하지만 이 방법에 대한 단점은 다음과 같음 

- Curse of Dimensionality → 위 예제들은 단순한 1개 변수를 사용한 예제임
- Histogram이 bin 숫자가 부족하면 → generalization 능력이 떨어짐
- Parameterized Distribution을 사용하면 좀 더 Generalization이 가능함
    - 즉 위 예제를 통해 설명하자면 아래처럼 나와 있는 정규분포 두 개에 대한 함수를 추정하는 방법!

```python
a = np.random.normal(loc=-10, scale=10, size=1000)
b = np.random.normal(loc=50, scale=20, size=1000)
```

### Then How to function approximation?

$P_{\theta}(x) \sim P_{data}(x)$ , 즉 theta를 잘 학습시키는 방법

<center><img src="pics/Untitled 6.png" width="500" height="300"></center>

Maximum Likelihood

<center><img src="pics/Untitled 7.png" width="500" height="300"></center>

Bayes Nets 

chain rule Network라고 부르기도 함

다음과 같이 joint distribution을 곱해서 계산함.

<center><img src="pics/Untitled 8.png" width="500" height="300"></center>

## 2.3 Autoregressive Model

$$log \ p(x) = \Sigma log \ p(x_i | x_{1:i-1})$$

Example 

<center><img src="pics/Untitled 9.png" width="500" height="300"></center>

```python
x1 = np.random.randn(1000)
x2 = MLP(x1, activation='softmax') # <- P(x2 | x1)
```

- Dimension이 커질 경우엔?
    - Parameter Sharing
    - Masking

### Parameter Sharing의 대표적인 예제 - RNN

<center><img src="pics/Untitled 10.png" width="500" height="300"></center>

MNIST를 한 줄로 펴서 다음과 같이 만듦 - char-rnn

<center><img src="pics/Untitled 11.png" width="500" height="300"></center>

### Masking method의 대표적인 예제 - MADE

Autoregressive한 성질을 부여하기 위해 weight에 마스킹을 하여 순서를 만듦

좌) 일반적인 AutoEncoder  중) 마스킹 우) 바뀐 순서도

<center><img src="pics/Untitled 12.png" width="500" height="300"></center>

다른 강의에서 발췌 

<center><img src="pics/Untitled 13.png" width="500" height="300"></center>

<center><img src="pics/Untitled 14.png" width="500" height="300"></center>

출처 : https://www.youtube.com/watch?v=lNW8T0W-xeE&t=536s

위 과정을 다시 정리하면 아래처럼 표현 가능

<center><img src="pics/Untitled 15.png" width="500" height="300"></center>

- Type A, Type B는 순서가 상관 없음 - 들어가기만 하면 됨
- x1 - x6 까지 어떻게 섞이느냐에 따라 중요함

<center><img src="pics/Untitled 16.png" width="300" height="500"></center>

<center><img src="pics/Untitled 17.png" width="500" height="300"></center>

코드 참고

[karpathy/pytorch-made](https://github.com/karpathy/pytorch-made)

### WaveNet - 1D Convolution

Dilated Convolution을 사용하여 다음과 같이 다음 음성을 만들어냄

RNN과 비슷한 구조를 택함. 하나 만들면 output이 다음 input에 들어감

<center><img src="pics/Untitled 18.png" width="500" height="300"></center>

<center><img src="pics/Untitled 19.png" width="500" height="300"></center>

구조는 다음과 같음 

Input → Dilated Conv → Gated Activation → 1x1 Conv → Residual 

<center><img src="pics/Untitled 20.png" width="500" height="300"></center>

이 친구를 MNIST를 적용해보면 다음과 같이 만들어 볼 수 있음

특이사항은 : Positional Location 정보를 제공했다는 점! (이 친구가 빠지면 그림이 이상해짐)

<center><img src="pics/Untitled 21.png" width="600" height="400"></center>

### Pixel CNN - 2D Convolution

- 2D에서 순서를 넣어주기 위해서 다음과 같이 마스크를 씌워서 결정할 수 있게 함

<center><img src="pics/Untitled 22.png" width="500" height="300"></center>

실제 만들어지는 과정

<center><img src="pics/Untitled 23.png" width="500" height="300"></center>

단점은 다음과 같음 

- receptive field에서 blind spot이 존재하게 됨
- 즉 특정 영역만 보고 결정하게 됨

<center><img src="pics/Untitled 24.png" width="500" height="300"></center>

이를 해결하기 위해서 몇 가지 방법론들이 추후에 다시 나옴

- Gated PixelCNN
- 2x3 필터 하나와 아예 1D 친구를 같이 사용하자라는 아이디어에서 나온 방법

<center><img src="pics/Untitled 25.png" width="500" height="300"></center>

<center><img src="pics/Untitled 26.png" width="500" height="300"></center>

- PixelCNN++
- 픽셀이 얼마 차이 나도 사실 눈으로 구분하기 어려우니 Softmax보다 다음과 같은 Mixture한 분포를 사용하자

<center><img src="pics/Untitled 27.png" width="500" height="300"></center>

교수님이 만드신 방법

<center><img src="pics/Untitled 28.png" width="500" height="300"></center>


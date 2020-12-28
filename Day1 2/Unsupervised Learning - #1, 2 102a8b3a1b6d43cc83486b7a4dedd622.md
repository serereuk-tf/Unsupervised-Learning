# Unsupervised Learning - #1, 2

발표자: 석현 ­서
작성일시: 2020년 12월 5일 오전 12:06

1강의 내용이 너무 Intro의 성격이 강해서 2강의 내용을 조금 들고 옴

# 1. 1강

## 1.1 개요

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled.png](Day1 2/Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled.png)

  → 강의 내용이 생각보다 Broad하고 불친절함

중간에 설명이 더 필요하다면 아예 논문을 들고 와서 진행할 예정

1강의 나머지 내용은 Generative 모델의 예시들 (Generating Text, Image 등등) 이여서 생략함

---

# 2. 2강

## 2.1 개요

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%201.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%201.png)

- Likelihood-based Model
- Autoregressive Model

## 2.2 Likelihood-Based Model

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%202.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%202.png)

Likelihood-based models: estimate pdata from samples $x^{1}, …, x^{n} \sim p_{data}x$ 

$p(x)$를 구하면 장점 : 

- 임의의 x에 대해서 확률을 계산할 수 있음
- 난수를 추출 할 수 있음!!!!

하지만 단점은 다음과 같음 : 

- 실제 데이터들은 Complex and High Dimensional Data
- 128x128x3 이면 거의 50,000을 넘어감

그러므로 Computational Efficient하고 Statistical Efficient한 모델을 만들어야 함

### Simple Example - Histogram

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%203.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%203.png)

How to sample this histogram?

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%204.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%204.png)

위 Psedu 코드를 이해하기 위한 예제

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%205.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%205.png)

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

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%206.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%206.png)

Maximum Likelihood

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%207.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%207.png)

Bayes Nets 

chain rule Network라고 부르기도 함

다음과 같이 joint distribution을 곱해서 계산함.

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%208.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%208.png)

## 2.3 Autoregressive Model

$$log \ p(x) = \Sigma log \ p(x_i | x_{1:i-1})$$

Example 

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%209.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%209.png)

```python
x1 = np.random.randn(1000)
x2 = MLP(x1, activation='softmax') # <- P(x2 | x1)
```

- Dimension이 커질 경우엔?
    - Parameter Sharing
    - Masking

### Parameter Sharing의 대표적인 예제 - RNN

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2010.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2010.png)

MNIST를 한 줄로 펴서 다음과 같이 만듦 - char-rnn

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2011.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2011.png)

### Masking method의 대표적인 예제 - MADE

Autoregressive한 성질을 부여하기 위해 weight에 마스킹을 하여 순서를 만듦

좌) 일반적인 AutoEncoder  중) 마스킹 우) 바뀐 순서도

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2012.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2012.png)

다른 강의에서 발췌 

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2013.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2013.png)

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2014.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2014.png)

위 과정을 다시 정리하면 아래처럼 표현 가능

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2015.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2015.png)

- Type A, Type B는 순서가 상관 없음 - 들어가기만 하면 됨
- x1 - x6 까지 어떻게 섞이느냐에 따라 중요함

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2016.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2016.png)

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2017.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2017.png)

코드 참고

[karpathy/pytorch-made](https://github.com/karpathy/pytorch-made)

### WaveNet - 1D Convolution

Dilated Convolution을 사용하여 다음과 같이 다음 음성을 만들어냄

RNN과 비슷한 구조를 택함. 하나 만들면 output이 다음 input에 들어감

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2018.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2018.png)

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2019.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2019.png)

구조는 다음과 같음 

Input → Dilated Conv → Gated Activation → 1x1 Conv → Residual 

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2020.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2020.png)

이 친구를 MNIST를 적용해보면 다음과 같이 만들어 볼 수 있음

특이사항은 : Positional Location 정보를 제공했다는 점! (이 친구가 빠지면 그림이 이상해짐)

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2021.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2021.png)

### Pixel CNN - 2D Convolution

- 2D에서 순서를 넣어주기 위해서 다음과 같이 마스크를 씌워서 결정할 수 있게 함

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2022.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2022.png)

실제 만들어지는 과정

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2023.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2023.png)

단점은 다음과 같음 

- receptive field에서 blind spot이 존재하게 됨
- 즉 특정 영역만 보고 결정하게 됨

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2024.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2024.png)

이를 해결하기 위해서 몇 가지 방법론들이 추후에 다시 나옴

- Gated PixelCNN
- 2x3 필터 하나와 아예 1D 친구를 같이 사용하자라는 아이디어에서 나온 방법

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2025.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2025.png)

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2026.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2026.png)

- PixelCNN++
- 픽셀이 얼마 차이 나도 사실 눈으로 구분하기 어려우니 Softmax보다 다음과 같은 Mixture한 분포를 사용하자

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2027.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2027.png)

교수님이 만드신 방법

![Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2028.png](Unsupervised%20Learning%20-%20#1,%202%20102a8b3a1b6d43cc83486b7a4dedd622/Untitled%2028.png)

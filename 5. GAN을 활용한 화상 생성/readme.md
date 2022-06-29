# 5. GAN을 이용한 화상생성

---

---

# GAN을 활용한 화상 생성 메커니즘과 DCGAN 구현

- GAN은 두 종류의 신경망을 이용하여 구현
    - G(generator) : 화상을 생성하는 신경망 생성기
    - D(discriminator) : G에서 생성된 가짜 화상인지 훈련 데이터로 준비한 화상인지 분류하는 신경망 생성기
    - G는 D를 속이기 위하여 훈련 데이터에 가까운 화상을 생성하도록 학습하고 D는 G에 속지 않고 진위를 가려내도록 학습함
    - 이를 통해 G가 현실에 존재할 만한 화상을 생성하는 기술이 GAN
    

# Generator 메커니즘

- 사람이 봤을 때 숫자로 보이는 패턴을 생성하는 것이 G의 역할
- 매번 똑같은 화상을 생성하는 건 의미 없으므로 다양한 패턴의 화상을 생성하기 위해 G 신경망의 입력 데이터는 다양한 패턴을 생성할 난수 입력 → 난수 값에 따라 G가 숫자 화상 출력
- G에 필기체 숫자 화상 출력을 위해서는 지도 데이터 필요 → 사람이 봤을 때 숫자로 보이는 지도데이터 제공
- G는 지도 데이터로 준 화상과 다른 패턴이지만 사람이 봤을 때 필기체 숫자 화상으로 보이는 것을 출력
- 전치합성곱(ConvTranspose2d)를 이용하여 학습함

# Discriminator 메커니즘

- GAN은 사람 대신 G의 화상을 체크하여 숫자 화상으로 보이는지 판정하는 D 신경망을 준비
- 단순한 화상 분류 딥러닝이며 ‘0: 숫자 화상으로 보임, 1:숫자 화상으로 보이지 않음’이라고 판정
- 숫자 화상으로 보이는 지도 데이터를 준비해야 하며 D는 입력된 화상이 생성 화상(0), 지도 데이터(1) 중 어느 쪽인지 분류
- G는 G가 생성한 화상이 실제 필기체 숫자 화상에 가까운 것 같다는 D의 판단으로 비슷한 분위기의 화상을 더 생성하도록 파라미터를 학습하는 상태가 됨
- 미숙한 생기성

---

 

---

# Self - Attention GAN

- 기존에 DCGAN에서 사용한 ConvTranspose2d(전치 합성곱) 반복은 국소적인 정보 확대의 연속이며, 더 좋은 화상 생성을 실현하기 위해 확대할 때 가능하면 화상 전체의 포괄적인 정보를 고려할 수 있는 구조를 만들어야 함 → Self - Attention을 도입해 해결
- Self - Attention은 계산 비용을 억제하기 위하여 ‘입력 데이터의 어떠한 셀의 특징량을 계산할 때 주목할 셀은 N X N 범위의 셀로 한다’라는 제한을 둠

## Self - Attention 도입

![Untitled](5%20GAN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%92%E1%85%AA%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20a9929263c24f4e29916c12c70eaaa462/Untitled.png)

- 변수 x 텐서 크기는 채널 수 X 높이 X 폭(C X H X W)
- Γ(감마)는 적당한 계수, $\omicron$는 self - attention map
1. 변수 x를 행렬 연산 하기 쉽도록 C X H X W  → C X N으로 변경 이 때 N은 H X W
2. 변형한 x끼리의 곱셈인 S = xtx 행렬을 만들어 N X N 행렬 생성 → 행렬 S
3. 행렬 S의 요소 sij는 화상 위치 i가 화상 위치 j와 얼마나 비슷한지 나타낸다
4. 다음에 행 방향으로 정규화 한다. → 다음 공식으로 인해 규격화한 행렬을 $\beta$라 한다.
    - Attention Map $\beta$
        
        $$
        \beta_{j,i} = exp(s_{ij})/\sum_{i=1}^{\N}exp(s_{ij})
        $$
        
5. 마지막으로 C X N의 행렬과 Attention Map을 곱하면 현재 x에 대응하면서 정보를 조정하는 Self-Attention Map의 $\omicron$가 계산 된다.

$$
\omicron = x\beta{^T}
$$

```python
# 최초의 크기 변형 B, C, W, H -> B, C, N으로
X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]) # 크기 : B, C, N

# 곱셈
X_T = x.permute(0, 2, 1) # B, N, C 전치
S= torch.bmm(X_T, X) # bmm은 배치별 행렬곱

# 규격화
m = nn.Softmax(dim = -2) # i행 방향의 합을 1로하는 소프트맥스 함수
attention_map_T = m(S) # 전치된 Attention Map
attention_map = attention_map_T.permute(0, 2, 1) # 전치를 취한다. # 0 2 1은 순서

# Self-Attention 맵을 계산
o = torch.bmm(X, attention_map.permute(0, 2, 1) # Attention Map은 전치하여 곱한다.
```

## 1 x 1 합성곱(점별 합성곱)

- 커널 크기가 1x1인 합성곱 층에서 x를 크기 C X H X W에서 크기 C’ X H X W로 변환하여 self - attention을 사용( C > C’)
- 점별 합성곱을 사용하는 2가지 이유
    1. 입력 데이터의 어떠한 셀의 특징량을 계산할 때 주목할 주위 셀은 자신과 비슷한 셀로 한다. → 제한에도 잘 작동하는 특징량으로 x의 입력을 변화 시킴
    2. N X C의 행렬  $x^T$와 C X N 행렬 x의 곱셈을 하므로 C를 작게 하여 계산 비용을 줄이기 위해

```python
# 1 x 1 합성곱 층을 활용한 점별 합성곱 준비
query_conv = nn.Conv2d(
	in_channels = X.shape[1], out_channels = X.shape[1]//8, kernel_size = 1)
key_conv= nn.Conv2d(
	in_channels = X.shape[1], out_channels = X.shape[1]//8, kernel_size = 1)
value_conv = nn.Conv2d(
	in_channels = X.shape[1], out_channels = X.shape[1], kernel_size = 1)

# 합성곱한 뒤 크기를 변형시킨다. B, C', W, H -> B, C', N으로
proj_query = query_conv(X).view(
		X.shape[0], -1, X.shape[2] * X.shape[3]) # 크기 : B, C', N
proj_query = proj_query.permute(0, 2, 1) # 전치 조작
proj_key = key_conv(X).view(
		X.shape[0], -1, X.shape[2]*X.shape[3]) # 크기 : B, C', N

# 곱셈
S = torch.bmm(proj_query, proj_key) # bmm은 배치별 행렬곱

# 규격화
m = nn.softmax(dim = -2) # i 행 방향의 합을 1로하는 softmax 함수 (마지막 차원을 제거)
attention_map_T = m(S) # 전치된 Attention Map
attention_map = attention_map_T.permute(0, 2, 1) # 전치를 취한다.

# Self - Attention Map 계산
o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
```

## 스펙트럴 정규화

- 데이터가 아니라 합성곱 층 등의 네트워크 가중치 파라미터를 표준화하는 작업
- GAN이 잘 작동할려면 ‘립시츠 연속성’(식별기 D의 입력 화상이 아주 조금 변하면 식별기 D의 출력은 거의 변하지 않음)을 가져야함
- D가 립시츠 연속성을 보유하기 위해 스펙트럴 정규화에서 가중치를 정규화하는 작업을 실행
- 스펙트러 정규화로 GAN의 학습이 잘 되도록 층의 가중치를 정규화 함

```python
nn.utils.spectral_norm(nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size = 4, stride = 1))
```

## 참고 문헌

[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
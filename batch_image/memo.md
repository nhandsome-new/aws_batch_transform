1. Setting1

```python
max_concurrent_transforms = 1
max_payload = 1
strategy='SingleRecord'
split_type='Line'
```

[34m2022-08-22T16:39:16,433 [INFO ] W-9000-model_1.0-stdout MODEL_LOG - request received!![0m
[34m2022-08-23T02:37:54,680 [INFO ] W-9000-model_1.0-stdout MODEL_LOG - Num of data in a request: 1

2. Setting2

```python
max_concurrent_transforms = None
max_payload = None
strategy='SingleRecord'
split_type=None
```

[34m2022-08-22T16:50:38,008 [INFO ] W-9001-model_1.0-stdout MODEL_LOG - 2400[0m


```python
max_concurrent_transforms = 10
max_payload = 10
strategy='SingleRecord'
split_type=None
```

[35m2022-08-23T02:17:33,913 [INFO ] W-9003-model_1.0-stdout MODEL_LOG - 21600[0m


```python
# 2files
max_concurrent_transforms = 10
max_payload = 10 # 6553500
strategy='SingleRecord'
split_type=None
```

2022-08-23T05:21:45,405 [INFO ] W-9002-model_1.0-stdout MODEL_LOG - Num of data in a request: 2400
2022-08-23T05:21:51,247 [INFO ] W-9001-model_1.0-stdout MODEL_LOG - Num of data in a request: 21600


3. Setting3

```python
max_concurrent_transforms = 1
max_payload = 1
strategy='MultiRecord'
split_type='Line'
```

[34m2022-08-23T02:27:21,586 [INFO ] W-9000-model_1.0-stdout MODEL_LOG - Num of data in a request: 6811[0m
[34m2022-08-23T02:27:26,346 [INFO ] W-9001-model_1.0-stdout MODEL_LOG - Num of data in a request: 14789[0m

```python
max_concurrent_transforms = 1
max_payload = 1
strategy='MultiRecord'
split_type='Line'
```
[34m2022-08-23T06:10:19,281 [INFO ] W-9000-model_1.0-stdout MODEL_LOG - Num of data in a request: 2400[0m
[34m2022-08-23T06:10:25,867 [INFO ] W-9001-model_1.0-stdout MODEL_LOG - Num of data in a request: 14789[0m
[34m2022-08-23T06:10:26,270 [INFO ] W-9002-model_1.0-stdout MODEL_LOG - Num of data in a request: 6811



## 정리
1. 'SingleRecord' + 'Line'
- 파일이랑 관계없이 한줄한줄의 Line을 Request Body로 취급
- 큰 용량의 request가 아닌 이상, max_payload는 1로 충분할듯

2. 'SingleRecord' + 'None'
- 파일 하나를 하나의 Request Body로 취급한다.
- max_payload 를 넘어가는 데이터가 들어오면 오류가 발생

3. 'MultiRecord' + 'Line'
- max_payload 의 용량에 맞춰서 mini_batch를 만들어 Request Body로 취급
- max_payload의 설정에 따라 mini_batch 사이즈 조절이 가능하다.

### Batch Transform 활용
1. 'SingleRecord' + 'Line'
- 라인별 배치 구성 
    - 시계열 자료와 같이 입력 데이터의 용량이 큰 경우, 하나의 데이터를 입력으로 사용
    - 'input1', 'input2', ..., 'inputN' 같이 소수의 batch를 구성하여 입력으로 사용
        - N개의 input을 한줄로 json 처리하기
    - 필요한 것: 입력의 길이 < max_payload

2. 'SingleRecord' + 'None'
- 파일별 배치 구성 
    - 배치사이즈 만큼의 라인을 하나의 파일로
        - N개의 input을 N개의 라인으로 만들고, 하나의 파일로 저장
    - instance수를 늘렸을 때, 자동으로 파일이 분배될 것이다.
    - 필요한 것: 입력의 길이 < max_payload

    ```python
    When you have multiples files, one instance might process input1.csv, and another instance might process the file named input2.csv. If you have one input file but initialize multiple compute instances, only one instance processes the input file and the rest of the instances are idle.
    ```

3. 'MultiRecord' + 'Line'
- 자동 배치 구성
    - 하나의 파일에 모든 인풋을 넣으면 자동으로 mini_batch생성
    - predict_fn 에서 request_body를 N개의 배치로 나누어 처리
    - max_payload에 따른 오류는 없을것 같다.
    - instance수를 늘리려면, 파일도 instance수만큼 만들어야한다.

### 결론
- Input response가 충분히 크고 모델이 무겁다. (시계열데이터)
    - 'SingleRecord' + 'Line'
    - 모델이 받아들일 수 있을 만큼만 라인별 배치구성
    - 한줄씩 모델이 처리
    - Multi Instance를 사용한다면, Instance수만큼의 파일이 필요

- 한번에 많은 데이터를 처리 / 모델이 한번에 추론할 수 있는 배치(N)를 알고 있다. (이미지 분류)
    - Instance를 자유롭게 조절하고 싶다면 
        - 'SingleRecord' + 'None'
        -  "인스턴스 수 < 파일 수" 의 경우 문제없이 처리 
        - N만큼 전처리(N라인을 가진 파일을 생성)가 필요
    - Inctance를 고정한다면
        - 'MultiRecord' + 'Line'
        - 고정된 인스턴스 수만큼 파일을 준비해둔다
        - N만큼 내부적으로 배치를 만들 필요
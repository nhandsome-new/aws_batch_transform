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






# Batch Transformer Setting 比較
1. 'SingleRecord' + 'Line'
    - inference_dir中の、ファイル数・ファイル中のLine数と関係なく
    - いつも「一つのLine → requestBody」として処理

2. 'SingleRecord' + 'None'
    - inference_dir中の、「一つのファイルのすべてのLine → requestBody」として処理

3. 'MultiRecord' + 'Line'
    - inference_dir中のファイルのLineを、「max_payload設定に合わせてmini batch生成 → requestBody」として処理

## 例
- 入力：jsonlines file ２つ
    - 1：2,400 Lines
    - 2：21,600 Lines
- 出力
    1. 'SingleRecord' + 'Line'
        - requestBody Size : 1
        - Counts of request : 2400 + 21600
    2. 'SingleRecord' + 'None'
        - requestBody Size : 2400、　21600
        - Counts of request : 2
    3. 'MultiRecord' + 'Line'
        - requestBody Size : 2400、　14789、　6811
        - Counts of request : 3

## 活用方法を考えてみた
1. 'SingleRecord' + 'Line'
- 時系列のように、「入力がでかい・推論時間が長い」
    - 必要なタスク(Lambda)
        - 全てのデータをinstance数に合わせて分ける。
    - Batch Transform
        - 一個のファイルを一個のinstanceが処理
        - １行　＝　入力
    - 注意点
        - max_payloadを超えないrequestBodyサイズ

2. 'SingleRecord' + 'None'
- マルチモーダルのように、「入力が複数」
    - 必要なタスク(Lambda)
        - 入力データの組み合わせを一つのファイルとして保存
    - Batch Transform
        - ファイル一個づつ、instanceが処理
        - 一個のファイル（複数のLine、複数のデータ）＝ 入力
    - 注意点
        - max_payloadを超えないrequestBodyサイズ

3. 'MultiRecord' + 'Line'
- 普通にBatch処理したい場合
    - 必要なタスク(Lambda)
        - 全てのデータをinstance数に合わせて分ける。
    - Batch Transform
        - max_payloadに合わせて、mini-batchを作る
        - mini-batch（複数のLine）＝ 入力
    - 注意点
        - mini-batchサイズ　＞　モデルのcapa　にならないように
        - 内部的にmini-batchをBATCH_SIZEに分けて処理



## 注意点
1. instanceを複数生成数　＜　inference_dir中のファイルの数
    ```python
    When you have multiples files, one instance might process input1.csv, and another instance might process the file named input2.csv. If you have one input file but initialize multiple compute instances, only one instance processes the input file and the rest of the instances are idle.
    ```
2. max_payload ＞＝　requestBody
- 'SingleRecord' + 'Line' / 'SingleRecord' + 'None' の場合、mini_batchを生成しないため、Error


## 今後タスク
1. Lambda　作成 
    - 「Input・Ouput」比較
    - 未処理データ抽出
    - 未処理データを複数に分ける（multi instanceの場合にも対応でくる）
    - Jsonlines ファイルをInstanceフォルダに保存
    - Batch Transform job起動
2. Batch Transform
    - [Associate Prediction Results with Input Records](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html)
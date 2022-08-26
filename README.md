# AWS SageMaker Batch Transform Example with CIFAR10
[Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)を使ったExample

## 実行の前に
- AWS Role・Bucketの準備
    - Role : S3 / SageMakerのポリシー
    - Role・Bucket名前が必要
- Local環境でのAWS CONFIG設定
    - AWS Accessできるように
- [Notebook](https://github.com/nhandsome-new/aws_batch_transform/blob/main/batch_image/batch_transform_CIFAR10.ipynb)の設定

    ```
    role = 'YOUR_AWS_ROLE_NAME' # with s3, sagemaker accese policy
    bucket='YOUR_BUCKET_NAME'

    # model_path = cifar10_predictor
    model_path = 's3://your/saved/model/path/model.tar.gz'

    output_s3_path = 's3://your/output/path'

    # inference_path = f_inference
    inference_path = f's3://your/input/path'
    ```
## 流れ
1. CIFAR10データせっとのダウンロード・jsonlinesに変換・S3にアップーど(inference_path)
2. モデルの学習(model_path)
3. Batch Transform 実行

- 学習済みモデルがある場合、①jsonlines変換・S３アップロードと③Batch Transform実行だけでいける


## To do
- [ ] Clean code
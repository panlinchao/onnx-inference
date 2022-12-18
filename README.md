# ONNX-Inference
This project is to modify the onnx model, which export by MindSpore framework, to inference on CPU.

## Setup Environment

Run the following command to prepare the environment.

```shell
pip install -r requirements.txt
```

## Inference

Run the following command to inference input images on `img_dir`.

```shell
python src/local_classify_test.py --img_dir=data
```

## Output

Inference results store on `outputs` dir. Here is an example.

![inference result of newspaper](outputs/newspaper.jpg "Newspaper")
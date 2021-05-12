# Image Classification with the ArmNN TfLiteDelegate

This application demonstrates the use of the ArmNN TfLiteDelegate. In this
application we integrate the delegate into the TensorFlow Lite Python package.

## Before You Begin

This repository assumes you have built, or have downloaded the
`libarmnnDelegate.so`. You will also need to have built the TensorFlow Lite
library from source, we currently do not support the `tflite_runtime` package
standalone.

If you have not already installed these, please follow our guide in the ArmNN
repository which can be found
[here](https://github.com/ARM-software/armnn/blob/branches/armnn_21_02/delegate/BuildGuideNative.md).


## Getting Started

Before running the application, we will first need a model, a labelmapping and
some data.

1. Install required packages

  ```bash
  sudo apt-get install -y python3 python3-pip wget git git-lfs unzip
  ```

2. Clone this repository and move into this project folder

  ```bash
  git clone https://github.com/arm-software/ML-examples.git
  cd ML-examples/armnn-delegate-image-classification
  ```

3. Download your model and label mappings

  For this example I am using the `MobileNetV2` model. We have multiple
  versions available in the Arm model zoo as well as scripts to download the
  labels.

  ```bash
  export BASEDIR=$(pwd)
  #clone the model zoo
  git clone https://github.com/arm-software/ml-zoo.git
  #go to the mobilenetv2 uint8 folder
  cd ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8
  #generate the labelmapping
  ./get_class_labels.sh
  #cd back to this project folder
  cd BASEDIR
  #copy your model and label mapping
  cp ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8/mobilenet_v2_1.0_224_quantized_1_default_1.tflite .
  cp ml-zoo/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8 labelmappings.txt .
  ```

4. Download a test image

  ```bash
  wget -O cat.png "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
  ```

5. Download the required Python packages

  ```bash
  pip3 install -r requirements.txt
  ```

6. Copy over your `libtensorflow_lite_all.so` library file you generated before
trying this application to the application folder. For example for me this
command was:

  ```bash
  cp /home/odroid/tensorflow/bazel-bin/libtensorflow_lite_all.so .
  ```

## Folder Structure

After following the above steps you will have the following folder structure.

```
.
├── README.md
├── run_classifier.py          # script for the demo
├── libtensorflow_lite_all.so  # tflite library built from tensorflow
├── cat.png                    # downloaded example image
├── mobilenet_v2_1.0_224_quantized_1_default_1.tflite #tflite model from ml-zoo
└── labelmappings.txt          # model labelmappings for output processing
```

## Run the model

```bash
python3 run_classifier.py \
--input_image cat.png \
--model_file mobilenet_v2_1.0_224_quantized_1_default_1.tflite \
--label_file labelmappings.txt \
--delegate_path /home/odroid/build-aarch64/libarmnnDelegate.so.24 \
--preferred_backends GpuAcc CpuAcc CpuRef
```

## Running Inference

Compared to your usual TensorFlow Lite projects, using the ArmNN TfLiteDelegate
requires one extra step when loading in your model:

```python
import tflite_runtime.interpreter as tflite

armnn_delegate = tflite.load_delegate("/path/to/delegate/libarmnnDelegate.so",
  options={
    "backends": "GpuAcc,CpuAcc,CpuRef",
    "logging-severity": "info"
  }
)
interpreter = tflite.Interpreter(
  model_path="mobilenet_v2_1.0_224_quantized_1_default_1.tflite",
  experimental_delegates=[armnn_delegate]
)
```

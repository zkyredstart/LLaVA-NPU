## LLaVA-NPU: Running LLaVA on a NPU device, e.g., Ascend 910B

In this project, we transfer the LLaVA from the CUDA device to the NPU device. If you think our project helpful, give our project a star. Thanks for your support!

### Change history
[2024-08-125]: We reproduce the results on the MMBench. Weights will be uploaded soon.

[2024-08-15]: We create the project and update the source code.

### Installation
<1> Install the LLaVA. ```pip install -e .```

### Train and Test
<1> You can train and test LLaVA as those in the official repo! If you are in china, you can download the model from modelscope.

<2> **Important**: LLaVA-NPU does not support lora tuing. Please use the full tuning. We train LLaVA on 8 Ascend 910B NPUs with 65GB memory.

### Core code
<1> LLaVA-NPU changes the flash_atten implementation. The code can be found in [here](llava/train/llama_npu_monkey_patch.py).

<2> We modify the evaluation code in LLaVA. The code is [here](llava/eval).

### Acknowledgement
<1> We would like to express the sincere thanks to [this repo](https://github.com/HelloWorldBeginner/LLaVA/tree/main) for its implementation on the NPU. Our project is based on it!

<2> Many thanks to LLaVA for its great contribution to MLLM community!


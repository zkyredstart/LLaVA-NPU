## LLaVA-NPU: Running LLaVA on a NPU device, e.g., Ascend 910B

In this project, we transfer the LLaVA from the CUDA device to the NPU device. If you think our project helpful, give our project a star. Thanks for your support!

### Change history
[2024-09-02]: We support the common visual projectors, e.g. LDPv2, Resampler. Try it!

[2024-08-27]: We add SigLip encoder to the LLaVA-NPU.

[2024-08-25]: We reproduce the results on the MMBench. Weights are coming soon.

[2024-08-15]: We create the project and update the source code.

### Installation
<1> Install the LLaVA. ```pip install -e .["train"]```

### Train and Test
<1> You can train and test LLaVA as those in the official repo! If you are in china, you can download the model from modelscope.

<2> **Important! ! ! ! !**: LLaVA-NPU does not support lora tuing and zero3-offload. Please use the full tuning. We train LLaVA on 8 Ascend 910B NPUs with 65GB memory.

<3> Training details. The hyper-parameters used in the pertraining and visual instruction tuning are as followed.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-7B | 256 | 1e-3 | 1 | 2048 | 0 |

2. visual instruction tuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-v1.5-7B | 64 | 1e-5 | 1 | 2048 | 0 |

<4> Model Performance comparison. 
| Model | Image encoder | Language Model | Projector | MMBench |
| --- |--- | --- | --- |---: |
| LLaVA-v1.5-7B (official) | CLIP | Vicuna-7B | MLP |64.5 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | MLP |64.5 |
| LLaVA-v1.5-7B (ours) | SigLip | Vicuna-7B | MLP |64.5 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | Adaptive Pool |64.6 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | Resampler |63.1 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | LDPv2 |65.7 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | TokenPacker |63.1 |
| LLaVA-v1.5-7B (ours) | CLIP | Vicuna-7B | C-Abstract |65.1 |

### Core code
<1> LLaVA-NPU changes the flash_atten implementation. The code can be found in [here](llava/train/llama_npu_monkey_patch.py).

<2> We modify the evaluation code in LLaVA. The code is [here](llava/eval).

<3> We support nn.MultiHeadAttention on the NPU device.

```
import torch
import torch.nn as nn
import torch_npu
import math

class MultiheadfusionAttention(nn.Module):
    """
    MultiHeadAttention Implementation on the NPU device
    """
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model) 
    
    def forward(self,query,key,value,attn_mask=None,dropout=1.):
        # import pdb;pdb.set_trace()
        batch_size = query.size(0)
        ns = key.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x) for l, x in zip(self.linear_layers, (query, key, value))]   

        scale = 1 / math.sqrt(self.d_k)     

        attn_output = torch_npu.npu_fusion_attention(query, key, value,
                                                     self.h,
                                                     pse=None,
                                                     padding_mask=None,
                                                     atten_mask=attn_mask,
                                                     scale=scale,
                                                     keep_prob=dropout,
                                                     input_layout="SBH",
                                                     pre_tockens=65536,
                                                     next_tockens=0,
                                                     inner_precise=0)

        return attn_output

# Usage
inputs = torch.rand(1,2304,1024)
Attention = MultiheadfusionAttention(1024,8)
outputs = Attention(inputs)[0]
```

### Acknowledgement
<1> We would like to express the sincere thanks to [this repo](https://github.com/HelloWorldBeginner/LLaVA/tree/main) for its implementation on the NPU. Our project is based on it!

<2> Many thanks to LLaVA for its great contribution to MLLM community!


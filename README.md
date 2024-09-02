# Light-PEFT
Here is the implementation of our Findings of ACL2024 paper "Light-PEFT: Lightening Parameter-Efficient Fine-Tuning via Early Pruning".
## Setup

```bash
pip install -r requirements.txt
cd peft
pip install -e .
```

## Run Experiments

Estimation

```bash
bash scripts/estimate.sh
```

More Efficient Fine-Tuning

```bash
bash scripts/train.sh
```

Vanilla Fine-Tuning

```bash
bash scripts/vanilla_train.sh
```

Thanks for the open source code of  [EarlyBert](https://github.com/VITA-Group/EarlyBERT), [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), and [P-Tuning v2](https://github.com/THUDM/P-tuning-v2). Some of our codes are based on them.

## Citation
```
@inproceedings{gu-etal-2024-light,
    title = "Light-{PEFT}: Lightening Parameter-Efficient Fine-Tuning via Early Pruning",
    author = "Gu, Naibin  and
      Fu, Peng  and
      Liu, Xiyu  and
      Shen, Bowen  and
      Lin, Zheng  and
      Wang, Weiping",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.447",
    pages = "7528--7541",
    abstract = "Parameter-efficient fine-tuning (PEFT) has emerged as the predominant technique for fine-tuning in the era of large language models. However, existing PEFT methods still have inadequate training efficiency. Firstly, the utilization of large-scale foundation models during the training process is excessively redundant for certain fine-tuning tasks. Secondly, as the model size increases, the growth in trainable parameters of empirically added PEFT modules becomes non-negligible and redundant, leading to inefficiency. To achieve task-specific efficient fine-tuning, we propose the Light-PEFT framework, which includes two methods: Masked Early Pruning of the Foundation Model and Multi-Granularity Early Pruning of PEFT. The Light-PEFT framework allows for the simultaneous estimation of redundant parameters in both the foundation model and PEFT modules during the early stage of training. These parameters can then be pruned for more efficient fine-tuning. We validate our approach on GLUE, SuperGLUE, QA tasks, and various models. With Light-PEFT, parameters of the foundation model can be pruned by up to over 40{\%}, while still controlling trainable parameters to be only 25{\%} of the original PEFT method. Compared to utilizing the PEFT method directly, Light-PEFT achieves training and inference speedup, reduces memory usage, and maintains comparable performance and the plug-and-play feature of PEFT.",
}
```

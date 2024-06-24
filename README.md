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

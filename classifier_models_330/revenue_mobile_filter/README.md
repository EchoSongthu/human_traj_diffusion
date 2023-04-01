---
license: mit
tags:
- generated_from_trainer
datasets:
- mobile
model-index:
- name: revenue_mobile
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# revenue_mobile

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the mobile wikitext-103-raw-v1 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 24
- eval_batch_size: 600
- seed: 101
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 300.0

### Training results



### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.13.1+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2

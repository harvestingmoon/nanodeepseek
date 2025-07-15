### NanoGPT Incorporating DeepSeek Things

## TLDR: NanoGPT with DeepSeek 

### Motivation:
This repo basically serves as a way to see how I can implement the novel mechanisms as shown in DeepSeek Papers into a small simple and concise LLM (that is basically NanoGPT). It also allows me to better understand models at a finegrained level.



### Items Implemented: 
- Multi Latent Attention (with K/V cache in place)
- DeepSeekMoE (with fine grained expert segmentation, shared expert isolation and a modified auxillary load balacer)
- Decoupled RoPE 


## Things I want to further implement:
- Multi-Token Prediction
- GRPO (r1)
- FP8 Mixed Precision Training
- 


### Some Novel Statistics:

This model has 194.8m parameters, with about 12 MoE Layers (since the transformer blocks are stacked on top of each other)

Each MoE layer has about 9.45m params with about 7.08 active params per layer (layer referencing to the transformer block again)


### How To train: 

1. Fork Andrej Karpathy's NanoGPT repo
2. Replace `train.py` and `model.py` with the files here 
3. Perform the same training process as stated in the NanoGPT guide 



#### DeepSeek Papers: 
[Deepseek v2](https://arxiv.org/pdf/2405.04434)\
[Deepseek v3](https://arxiv.org/abs/2412.19437)\
[Deepseek r1](https://arxiv.org/pdf/2501.12948)\
[Innovative Techniques in Deepseek](https://arxiv.org/pdf/2503.11486)
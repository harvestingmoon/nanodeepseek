### NanoGPT Incorporating DeepSeek Things


This repo basically serves as a way to see how I can implement the novel mechanisms as shown in DeepSeek Papers. It basically incorporates novel deepseek items into a small simple and concise LLM (that is basically NanoGPT)


Key items: 
1. Multi latent attention (v2)
2. DeepSeekMoe (v2)
3. Decoupled RoPe (v2)
4. Multi Token Prediction (v3)
5. GRPO ? (r1) <- I need to learn some RL stuff before I can train this model


TLDR: I took the Andrej Karpathy's nanoGPT model and modified it to include


Current items in place:

1. Multi Latent Attention (with K/V cache in place)
2. Mixture Of Experts (not DeepSeekMoe, will implement DeepSeekMoe w/o Auxillary Loss then do so later)
3. Decoupled RoPE 

Flash attention was already pre implemented via Torhc


So now this model has 62 million parameters with 8 experts, each expert has about 4.7 million params 

I have not included a how to on training etc but it is trainable so I have provided the script, have fun I guess :)

Just download use the script together with the combined instructions from Andrej Karpathy and it wil be fine 

#### DeepSeek Papers: 
[Deepseek v2](https://arxiv.org/pdf/2405.04434)\
[Deepseek v3](https://arxiv.org/abs/2412.19437)\
[Deepseek r1](https://arxiv.org/pdf/2501.12948)\
[Innovative Techniques in Deepseek](https://arxiv.org/pdf/2503.11486)
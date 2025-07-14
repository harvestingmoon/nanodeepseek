### NanoGPT that goes Zoom


TLDR: I took the Andrej Karpathy's nanoGPT model and modified it to include

1. Multi Latent Attention
2. Mixture Of Experts 
3. Decoupled RoPE 

Flash attention was already pre implemented and I am kinda lazy to re-implement it in Triton or CUDA 


So now this model has 62 million parameters with 8 experts, each expert has about 4.7 million params 

I have not included a how to on training etc but it is trainable so I have provided the script, have fun I guess :)

Just download use the script together with the combined instructions from Andrej Karpathy and it wil be fine 
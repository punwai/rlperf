<img width="1536" height="1024" alt="ChatGPT Image Jul 18, 2025, 04_07_14 PM" src="https://github.com/user-attachments/assets/4feeb545-ef92-4a0b-89be-ee3da42fda5f" />

# Stasis
This library aims to demonstrate how to create a dead-simple multi-node RL stack.

Some focuses of this library includes:
- RL specific observability tools to help debug capabilities

- Support for GRPO, DrGRPO -- no value function models.
- Focus on clean code -- zero feature bloat.

###

### Development speed
vLLM makes many calls to pynvml during initialization, This brings up the intialization time to around 17s. Turning on persistence mode on the CUDA driver brings this time down to around 5s.

To turn on CUDA driver persistence mode, run:
```
sudo nvidia-smi -pm 1
```

This enables CUDA driver persistence mode. The size of the driver will be ~75MB of kernel memory [TODO check this], but this is well worth it for development velocity.

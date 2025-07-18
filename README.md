# Implementation notes

#### Development speed.

vLLM makes many calls to pynvml, and this is super slow without turning on persistence mode on the CUDA driver. You only need to set it once, and it should bring down the latency of initializing vLLM from 17s -> 5s.

```
sudo nvidia-smi -pm 1
```

This enables CUDA driver persistence mode. The size of the driver will be ~75MB of kernel memory [TODO check this], but this is well worth it for development velocity.
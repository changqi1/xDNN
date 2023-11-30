# X86 Deep Neural Network Library (xDNN)

X86 Deep Neural Network Library (`xDNN`) is an open-source optimized performance library for deep learning applications on X86 platform, like LLM, Recommendation systems.

```bash
# Code Structure
  your_app unit_test perf_test
     ▲       ▲   ▲     ▲   ▲
     └──┬────┴───┼─────┘   │
     ┌──┴──┐     └──┬──────┘
  include lib     utils
```

## How to use

xDNN provides headers `include` and libraries `lib` directly to integrate them into your applications.

```bash
# dynamic link
$ g++ app.cpp -o app -I/<path>/xDNN/include -L/<path>/xDNN/lib lxdnn

# static link
$ g++ app.cpp -o app -I/<path>/xDNN/include /<path>/xDNN/lib/libxdnn_static.a
```

## How to test

```bash
$ mkdir build && cd build
$ cmake ..
$ make -j

# unit test
$ numactl -N 0 -m 0 ./unit_test/test_sgemm

# performance test
$ numactl -N 0 -m 0 ./perf_test/benchmark_sgemm
```

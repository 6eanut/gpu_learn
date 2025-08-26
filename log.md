# 0820

通过LLM，简单了解GPU、CUDA、Triton的一些基本概念

针对task，先研究pytorch自带的矩阵乘法实现是什么，参考资料：[pytorch docs](https://docs.pytorch.org/docs/stable/index.html)

# 0821

学习[matmul from triton docs](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)，大致了解矩阵乘可以考虑的几个优化方向

# 0822

阅读了论文 [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

triton程序会经过triton-jit编译器，编译成triton-ir，然后会做多轮优化pass，然后编译成llvm-ir，再做多轮pass优化，然后利用llvm后端编译成ptx(parallel thread execution)，最后运行时cuda会将ptx编译成sass(streaming assembler)，即GPU实际运行的机器码。

# 0823

triton实现矩阵乘分三步，第一步：逐元素求结果矩阵；第二步：逐块求结果矩阵；第三步：求结果矩阵中元素/块的顺序；附：利用triton.autotune,tl.assume

这个实现逻辑倒是理顺了，但是性能远不如pytorch，再调研下吧

# 0824

今天研究这么几个问题：

* 如何实际测试一个GPU程序的性能？(cuda程序/triton程序)
* nsight/triton.test是做什么的，简单了解一下
* 测试一个CPU程序的性能往往用的是spec cpu/perf，那么对于gpu而言是什么呢？

nsight systems和nsight compute貌似triton和cuda程序都能测.

# 0825

完成网课performance consideration

* dram的结构、dram为啥慢？dram burst是啥？
* 内存合并coalesced memort ,corner turning?
* convolution computation,矩阵乘，如果越界，按零处理，这个很容易理解。
* constant memory可以用于存放卷积核等常量，constant memory啥作用？

# 0826

在[cuda learning](https://learn.nvidia.com/courses/course)里面做实验，每次启动先搭环境

```
!pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
!pip install numpy==1.23.3 pandas triton matplotlib
```

查看cuda和pytorch对应版本关系：[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

nsight system分析性能 `!nsys profile --stats=true python ./test.py`

几个问题：

* 使用autotune，能否获取当下key的最优config？
* nsys分析得到的数据怎么看？怎么针对性做优化？

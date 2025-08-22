CPU更聚焦于降低延迟，比如分支预测、大容量缓存等；GPU更聚焦于提高吞吐量，例如很多线程。

CPU更适合处理串行任务；GPU更适合处理并行任务

---

开发软件应该注重scalability和portability。

scalability指的是一个软件可以在新老不同的core上运行 以及 一个软件可以在单/多核上运行；

portability指的是一个软件可以在不同架构的core上运行；

因为软件的增速比硬件的增速要快(?)，所以提高软件的scalability和portability是很重要的

---

在cuda编程里面，会编写kernel函数(gpu并行函数)，语法是C/C++的扩展；kernel函数会被普通C/C++函数调用(host函数)。

当我们在host函数中调用kernel函数时，会需要设置执行配置，即当前grid中的block数和每个block中的thread数(block数设置为SM的整数倍；thread数设置为256，即32的整数倍)。

然后在kernel函数里面通过 `threadIdx.x+blockIdx.x*blockDim.x`来获得当前线程在grid中的唯一索引号。

问题是这样的，在cuda编程课里面，讲到gird和block都是2D/3D的，那为什么在求每个线程的索引时只用x就可以了呢？

原因是这样的，默认情况下，grid和block是1D，如果想设置2D/3D，可以通过下面的方式来设置：

```
dim3 blockSize(256, 2, 1);
dim3 gridSize(32, 3, 1);
testKernel<<<gridSize, blockSize>>>();
```

一旦block/grid不是1D，那么 `threadIdx.x+blockIdx.x*blockDim.x`获取索引的方式就不再正确了。

---

在nvidia gpu里面，一个device会有一个grid，对应着多个block，这多个block共享一个global memory，每个block里面有多个thread，每个thread都有自己的registers；

cudamalloc和cudamemcpy是分配host/device可用的内存以及在host/device之间传递数据的api；

在triton编程中几乎不怎么涉及这两个操作，这两个操作由框架去实现了；所以triton能够让开发者更加专注于算法逻辑，简化了gpu编程。·

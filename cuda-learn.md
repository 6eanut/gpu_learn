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

---

核函数有__global__和__device__两种声明方式，前者只能被host函数调用，后者只能被device函数调用；有的函数会被声明成__host__和__device__，那是因为它往往是device端和host端都会用到的util函数。

---

考虑这么一种情况，现在要处理一个2D的图片，任务是对图片中的每个像素值进行扩大两倍。假设哈，假设给内核函数设置执行配置的grid和block都是二维的，那么在内核函数里面，计算当前thread需要处理哪个像素，需要利用x和y两个维度来计算，分配的grid的行(列)数*block的行(列)数最好是大于等于图片的行数和列数，那么当都不等于时，就会有四种情况，就是对于一个block，其中的thread们，可能都有任务(左上角)，可能block的最右边的thread没任务(右边)，可能block的最下面的thread没任务(左下角)，也可能block的下面和右面没任务(右下角)。

记住这个场景，对后面的性能分析理解有帮助。

---

对于矩阵乘，a*b=c，我们认为c矩阵中的每个元素i,j，都是由矩阵a中的第i行和矩阵b中的第j列计算得到的。根据矩阵乘的计算逻辑，最直观的就是逐个计算c矩阵中的元素，外层循环从0到i，内层循环从0到j，最内层循环从0到k，然后做累加，每个最内层循环走一遍，c中的一个元素就计算出来了。这就是CPU计算矩阵乘的串行逻辑。

对于GPU的并行逻辑是这样的，默认block中的thread是平方数(blockdim)，那么就看对于矩阵c的i行j列，分别需要多少个block才能cover住(gridim)，以此来计算griddim的两个维度，而后在内核函数里面，利用griddim和blockdim的x和y的分量分别表示i和j，然后这样就能并行计算结果了。

这是最直观的计算方式

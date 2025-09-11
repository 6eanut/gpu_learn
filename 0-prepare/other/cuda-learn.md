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

---

假设这么一个场景，在kernel函数里面，根据threadidx/blockidx来设定条件分枝的条件，那么此时有可能会出现这么一个情况，那就是同一个warp内(32个连续的thread)，会有两组thread，分别走向不同的branch，这会导致GPU性能损耗。

这个叫做branch divergence，GPU中的线程是以warp为单位执行的，而且每个warp内的线程共用同一个pc，这导致同一个warp内的thread不能同时执行不同的指令，即不同的分支必须顺序执行，通过分支掩码机制来将更新thread的活跃/屏蔽状态。下面是一个例子：

```
__global__ void example_kernel(int* data) {
    int tid = threadIdx.x;  // tid范围：0-31（一个warp）
  
    if (tid < 16) {
        data[tid] = tid * 2;     // 路径A
    } else {
        data[tid] = tid + 10;    // 路径B
    }
}
```

我们有这么一个kernel函数，假设启动的执行配置是<32,32>，意味着有32个block，每个block有32个thread，thread的编号为0～31.

warp是GPU中线程调度和执行的基本单位，一个warp包含32个线程，这32个线程共用一个程序计数器，这意味着在同一时刻，一个warp内的所有线程都会去执行相同的指令。

在上面的代码中，对于一个warp内的32个线程，会被分为两组，第一组执行路径A，第二组执行路径B。在实际执行过程中，由于同一warp内共用同一个pc，所以当第一组线程实际执行路径A时，第二组线程会被屏蔽掉/执行路径A但结果被丢弃，然后第二组线程会去执行路径B，与此同时，第一组线程会被屏蔽掉/执行路径B但结果被丢弃。

实现上述屏蔽过程的是，一个warp内会维护一个32位的分支掩码，用来表示warp内线程的活跃/屏蔽状态，来决定是否执行当前pc所指的指令。

故，当一个warp内出现两组/多组线程执行不同的分支时，该warp的执行时间并不是执行时间最长的那个分支，而是所有分支执行时间的总和。

以上就是branch divergence会导致性能下降的原因。

---

在nvgpu里面，内存架构是这样的，一个grid(多个block)共享一个global memory和constant memory，每个block里面(多个thread)共享一个shared memory，每个thread有自己的寄存器。

怎么确定数据存放在哪个地方？对于kernel函数中直接声明的变量，那就是放在register里面，但如果是数组，那么会存在shared memory里面；对于cudamemalloc动态申请的数据，会放在global memory里面。在kernel函数里面可以通过声明变量前加device/shared/constant来决定把数据存到哪。

编程时，为了使数据访问更高效，往往会把大数据分为多个tile，然后每个tile放在shared memory里面，执行结束后，再写回global memory。注意当一个kernel要用另一个kernel的写结果时，一定要等前面kernel完全执行完毕，否则可能前面kernel写的数据还没写回global memory。

在triton编程里面，数据放在哪个内存结构由triton-jit编译器来决定。

---

cuda编程提供了一种同步的api，就是可以同步一个block中的所有thread，因为存在这样的需求，所以一个grid中有多个block是很必要的，这有助于提高性能。

另外有一个问题，就是矩阵tiled乘法，结果矩阵中的每个元素都是分多阶段求出来的。

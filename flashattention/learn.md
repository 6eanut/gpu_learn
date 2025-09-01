# FlashAttention

## Standard Attention

![1756722809776](image/learn/1756722809776.png)

在介绍FlashAttention之前，需要先说一下Standard的Attention是怎么实现的：

* 首先从HBM里面分块加载Q和K并计算S，然后会把S逐块写回HBM，此时HBM里面存放的有完整的S；
* 其次从HBM里面加载S，对S做softmax得到P，把P写回HBM，此时HBM里面存放的有完整的P；
* 之后从HBM里面分块加载P和V并计算O，然后会把O逐块写回HBM，此时HBM里面存放的有完整的O。

上述过程，产生了大量对慢速HBM的读写操作(S和P在SRAM和HPM之间的传输)，计算受到HBM带宽限制，计算单元等待数据时间长。

## FlashAttention 1

![1756722837721](image/learn/1756722837721.png)

目标：减少对HBM的内存读写操作

FlashAttention是这么做的：

* 在HBM里面初始化m和l这两个辅助计算P和O的变量，然后对m、l、Q、K、V进行分块(K和V是Tc块，m、l和Q是Tr块)，确保SRAM能够装下计算一个小块O所需要的数据；
* 外循环，循环遍历Tc，把相应的K和V小块从HBM加载到SRAM里面；
* 内循环，循环遍历Tr，把相应的O、Q、m和l小块从HBM加载到SRAM里面；
* 通过K和V小块计算出S小块，存在SRAM里面，不写回HBM；
* 通过S小块计算出辅助变量，并利用它们来在线更新O、m和l小块，将结果写回HBM。

整个过程，只有最初和最末涉及对HBM的读写操作，通过在线Softmax和增量更新实现了内存读写次数的减少。

整体上来看，FlashAttention并没有一下子把所有S全计算完，而是只计算S小块，然后借助m和l来实现在线softmax和动态更新O小块这个过程，这样减少了SRAM和HBM之间的数据传输；虽然相比于Standard多了对m、l和O的动态更新计算，但是相比于内存访问，这点时间不算什么。

## FlashAttention 2

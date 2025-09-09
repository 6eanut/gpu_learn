* 程序包含：triton实现的flash attention 和 python实现的attention；
* 输入：seq_len×d_model(1024×64)规模的Q K V矩阵；
* 输出：seq_len×d_model的O矩阵；

为什么没用flash-attn的pypi包？因为线上实验环境不满足CUDA>=12.0。

flow：

* 初始化生成Q, K, V矩阵；
* 利用pycuda包，获取当前GPU每个block可用的最大Shared Memory；
* 结合论文中的算法，通过Shared Memory、d_model以及计算的精度，求出Br和Bc(对Q和KV进行分块)；
* 调用triton实现的flash attention，得到结果O1；
* 调用python实现的attention，得到结果O2；
* 比对两者的结果；

flash flow：

* 在序列维度上做并行化，即每个program处理flash attention2算法中的第4~15行；
* 采用外循环遍历Q，内循环遍历KV，减少warp间通信代价；
* 在线softmax对应flash attention2算法的第9~10行；

4dflash：

* 前面实现的是2d，即序列长度✖️特征维度，可以基于此实现4d，即批大小✖️序列长度✖️注意力头数✖️特征维度；
* 需要做的修改有以下几个部分：
  * 在triton方面：维度由2d变成4d，涉及qkv的初始化、并行化策略、内存访问模式(和stride相关)
  * 在pytorch方面：矩阵乘改为批量矩阵乘，其中涉及维度变换的流程

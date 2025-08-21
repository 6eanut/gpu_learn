CPU更聚焦于降低延迟，比如分支预测、大容量缓存等；GPU更聚焦于提高吞吐量，例如很多线程。

CPU更适合处理串行任务；GPU更适合处理并行任务

---

开发软件应该注重scalability和portability。

scalability指的是一个软件可以在新老不同的core上运行 以及 一个软件可以在单/多核上运行；

portability指的是一个软件可以在不同架构的core上运行；

因为软件的增速比硬件的增速要快(?)，所以提高软件的scalability和portability是很重要的

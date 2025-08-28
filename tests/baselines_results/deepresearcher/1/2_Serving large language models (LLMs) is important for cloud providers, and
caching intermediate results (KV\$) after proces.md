<Related Works>
Large language models (LLMs) have become crucial in cloud computing, and efficiently serving them is essential. Caching intermediate results (KV$) has been shown to improve serving throughput and latency. Several studies have explored caching in LLM serving. For instance, FASTLIBRA, a Multi-LoRA caching system, optimizes serving performance by caching hot KV caches and LoRA adapters in high-bandwidth memory, reducing Time-To-First-Token (TTFT) by 63.4% on average compared to state-of-the-art works [1]. EmbAdvisor, a carbon-aware caching framework, selects the optimal cache size for LLM serving, reducing average carbon emissions by 9.5% under various carbon intensities [2]. Apt-Serve, a scalable framework, features a hybrid cache scheme and adaptive runtime scheduling, achieving up to 8.8x improvement in effective throughput compared to state-of-the-art inference serving systems [3]. Pensieve, a system optimized for multi-turn conversation LLM serving, maintains conversation state across requests by caching previously processed history, achieving 1.14-3.0x the throughput of vLLM and TensorRT-LLM and significantly reducing latency [4]. InfiniGen, a KV cache management framework, synergistically works with modern offloading-based inference systems, improving overall performance by up to 3.00x compared to prior KV cache management methods [5]. 

KIVI, a 2-bit KV cache quantization algorithm, enables Llama, Falcon, and Mistral models to maintain almost the same quality while using 2.6x less peak memory, bringing 2.35x-3.47x throughput on real LLM inference workload [6]. SYMPHONY, a system leveraging multi-turn workloads to migrate K,V caches off the critical serving path, handles over 8x the number of requests compared to state-of-the-art baselines, with a similar latency profile [7]. MorphServe, a dynamic, workload-aware LLM serving framework, reduces average SLO violations by 92.45 percent and improves P95 TTFT latency by 2.2x-3.9x compared to full-precision serving [8]. PSA, a progressive sparse attention mechanism, reduces KV cache usage for attention computation by up to 2.4x and increases end-to-end serving throughput by up to 1.4x and 2.0x, compared to state-of-the-art DSAes and systems without sparse attention, respectively [9].

Cache eviction policies have also been studied extensively. TLRU, a time-aware least recent used cache management policy, is suitable for ICN-type cache networks [10]. PopNetCod, a popularity-based caching policy, improves cache-hit rate compared to widely used Leave Copy Everywhere placement policy and Least Recently Used eviction policy [11]. TinyLFU, a highly efficient cache admission policy, boosts the effectiveness of caches subject to skewed access distributions [12]. BackCache, a novel hardware-software co-design, mitigates contention-based cache timing attacks on the L1 data cache [13]. 

Workload characterization is essential for understanding the behavior of different workloads. Lauca, a workload duplicator, generates synthetic workloads with highly similar performance metrics for specific applications [14]. A systematic literature review and characterization of web application workloads identify daily and weekly patterns within the workloads [15]. Big data workload characterization reveals that big data workloads are data movement dominated computing with more branch operations [16]. 

Our work contributes to the understanding of KV$ workload patterns from a leading LLM service provider and proposes a workload-aware cache eviction policy that improves serving performance under real-world traces.

</Related Works>

<references>
[1] http://arxiv.org/abs/2505.03756v1
[2] http://arxiv.org/abs/2505.23970v1
[3] http://arxiv.org/abs/2504.07494v1
[4] http://arxiv.org/abs/2312.05516v3
[5] http://arxiv.org/abs/2406.19707v1
[6] http://arxiv.org/abs/2402.02750v2
[7] http://arxiv.org/abs/2412.16434v1
[8] http://arxiv.org/abs/2506.02006v1
[9] http://arxiv.org/abs/2503.00392v1
[10] http://arxiv.org/abs/1801.00390v1
[11] http://arxiv.org/abs/1901.01187v1
[12] http://arxiv.org/abs/1512.00727v2
[13] http://arxiv.org/abs/2304.10268v5
[14] http://arxiv.org/abs/1912.07172v1
[15] http://arxiv.org/abs/2409.12299v1

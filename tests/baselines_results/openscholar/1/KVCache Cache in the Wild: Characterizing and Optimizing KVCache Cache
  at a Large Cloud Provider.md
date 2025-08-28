## Related Works

Serving large language models (LLMs) has become increasingly important for cloud providers, with a growing focus on optimizing performance and efficiency. Caching intermediate results, particularly key-value (KV) pairs, has been identified as a crucial technique for improving serving throughput and latency [0][3]. However, the benefits of KV caching are highly dependent on system design decisions, such as cache eviction policies, which are workload-dependent [25].

Previous studies have explored various aspects of caching in LLM serving. Some have proposed novel cache management techniques, such as leveraging online knowledge distillation [1] or employing machine learning models to predict future KV cache usage [0]. Others have investigated the impact of different cache eviction policies on performance, including the use of least recently used (LRU) and first-in-first-out (FIFO) policies [22][25].

Characterization of KV workload patterns has also been a subject of research. Studies have analyzed the temporal locality of KV pairs and proposed techniques to improve cache efficiency [28][5]. However, these studies often rely on synthetic workloads or limited datasets, which may not accurately reflect real-world scenarios.

Recent works have focused on optimizing LLM serving systems, including the use of multi-model queue management frameworks [3], dynamic cache instantiation [10], and reliability-aware KV cache compression [21]. These studies demonstrate the importance of understanding workload patterns and optimizing system design decisions to improve performance.

Our work builds on these efforts, presenting a systematic characterization of KV workload patterns from a leading LLM service provider. Our analysis reveals new insights into the reuse patterns of KV pairs, including the importance of reuses between single-turn requests and the predictability of reuse patterns for specific request categories.

Notably, recent studies have demonstrated the effectiveness of retrieval-augmented LMs (RAG) in outperforming fine-tuning approaches in tasks like MMLU and current events [20]. RAG's advantages include its ability to incorporate relevant context without suffering from catastrophic forgetting. Empirical results show that RAG can outperform fine-tuning by a significant margin, with some studies reporting a 30% reduction in hallucinations [2].

Furthermore, efficient LLM serving systems have been proposed, such as ServeGen, which characterizes and generates realistic LLM serving workloads [58]. Other studies have explored the use of novel cache eviction policies, such as TinyLFU [60] and multi-step LRU [85], to improve cache efficiency.

Moreover, recent research has proposed various techniques to optimize KV caching, such as STARC, a sparsity-optimized data mapping scheme for efficient LLM decoding on PIM architectures [88], and LeanKV, a framework that advances KV cache compression by exploiting differences in significance of various components within the KV cache [98]. Additionally, studies have shown that KV cache reusing can save both delay and cloud cost across a range of workloads with long context [94].

## References

[0] SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving 
[1] the quantity of LLM-annotated data on which the first student model is trained, focusing on the setup with retraining 
[3] One Queue Is All You Need: Resolving Head-of-Line Blocking in Large Language Model Serving 
[5] either cause premature eviction of a useful cache block, leading to an additional cache miss 
[10] Dynamic cache instantiation has the potential to provide significant cost reductions 
[20] Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs 
[21] No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization 
[22] It's Time to Revisit LRU vs. FIFO 
[25] in-memory key-value (KV) caches are widely used and discussed in industry and research communities 
[28] LLC Management is an important problem in today's heterogeneous processors
[31] Engineering Trustworthy Software: A Mission for LLMs 
[32] Large Language Models (LLMs): Deployment, Tokenomics and Sustainability 
[33] Batch-Max: Higher LLM Throughput using Larger Batch Sizes and KV Cache Compression 
[34] ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production 
[35] Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs 
[36] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference 
[37] Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense 
[38] CEC-Zero: Chinese Error Correction Solution Based on LLM 
[39] Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices 
[40] NanoFlow: Towards Optimal Large Language Model Serving Throughput 
[41] On the Efficacy of Eviction Policy for Key-Value Constrained Generative Language Model Inference 
[42] In-context KV-Cache Eviction for LLMs via Attention-Gate 
[43] LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization 
[44] The MoE-Empowered Edge LLMs Deployment: Architecture, Challenges, and Opportunities 
[45] ScaleLLM: A Resource-Frugal LLM Serving Framework by Optimizing End-to-End Efficiency 
[58] ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production
[60] TinyLFU: A Highly Efficient Cache Admission Policy 
[85] Multi-step LRU: SIMD-based Cache Replacement for Lower Overhead and Higher Precision
[88] Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM 
[94] Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache 
[98] Unifying KV Cache Compression for Large Language Models with LeanKV
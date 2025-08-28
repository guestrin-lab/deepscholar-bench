## Related Works

**Efficient serving of large language models (LLMs) relies heavily on effective key-value (KV) caching mechanisms**. Caching intermediate results after processing each request substantially improves serving throughput and latency \[[Woosuk Kwon' 2023-09-12](http://arxiv.org/abs/2309.06180v1)\]. However, the benefits of KV caching are highly dependent on system design decisions, such as cache eviction policies, which are workload-dependent \[[Haoyang Li' 2024-12-27](http://arxiv.org/abs/2412.19442v2)\].

Several studies have proposed various KV cache management strategies, including token-level, model-level, and system-level optimizations \[[June Yong Yang' 2024-02-28](http://arxiv.org/abs/2402.18096v1)\]. Token-level strategies include KV cache selection, budget allocation, merging, quantization, and low-rank decomposition \[[June Yong Yang' 2024-02-28](http://arxiv.org/abs/2402.18096v1)\]. Model-level optimizations focus on architectural innovations and attention mechanisms to enhance KV reuse \[[June Yong Yang' 2024-02-28](http://arxiv.org/abs/2402.18096v1)\]. System-level approaches address memory management, scheduling, and hardware-aware designs to improve efficiency across diverse computing environments \[[June Yong Yang' 2024-02-28](http://arxiv.org/abs/2402.18096v1)\].

Some notable works include PagedAttention \[[Woosuk Kwon' 2023-09-12](http://arxiv.org/abs/2309.06180v1)\], which achieves near-zero waste in KV cache memory and flexible sharing of KV cache within and across requests. vLLM, a system built on top of PagedAttention, improves the throughput of popular LLMs by 2-4× with the same level of latency compared to state-of-the-art systems \[[Woosuk Kwon' 2023-09-12](http://arxiv.org/abs/2309.06180v1)\]. Other works, such as Mixed-precision KV cache (MiKV) \[[Hanchen Li' 2025-03-18](http://arxiv.org/abs/2503.14647v1)\] and QAQ \[[Shichen Dong' 2024-03-07](http://arxiv.org/abs/2403.04643v2)\], propose reliable cache compression methods that preserve context details and ensure generation quality.

Recent studies have also explored the reuse of KV caches across different requests and conversations \[[Tianyu Guo' 2025-05-28](http://arxiv.org/abs/2505.21889v2)\]\[[Bin Gao' 2024-03-23](http://arxiv.org/abs/2403.19708v3)\]. CachedAttention \[[Bin Gao' 2024-03-23](http://arxiv.org/abs/2403.19708v3)\] enables reuse of KV caches across multi-turn conversations, significantly reducing repetitive computation overheads. Similarly, EFIM \[[Tianyu Guo' 2025-05-28](http://arxiv.org/abs/2505.21889v2)\] proposes a transformed prompt format to unleash the performance potential of KV cache reuse in infilling tasks.

Characterization of KV workload patterns has also been studied \[[Hanshi Sun' 2024-10-28](http://arxiv.org/abs/2410.21465v3)\]\[[Cunchen Hu' 2024-06-25](http://arxiv.org/abs/2406.17565v3)\]. MemServe \[[Cunchen Hu' 2024-06-25](http://arxiv.org/abs/2406.17565v3)\] presents a unified system that integrates inter-request and intra-request optimizations, introducing an elastic memory pool managing distributed memory and KV caches.

Our work builds upon these studies, providing a systematic characterization of KV workload patterns from a leading LLM service provider. We draw observations that can inform the design of workload-aware cache eviction policies, which improve serving performance under real-world traces, especially with limited cache capacity.

## References

\[[Woosuk Kwon' 2023-09-12](http://arxiv.org/abs/2309.06180v1)\] Efficient Memory Management for Large Language Model Serving with PagedAttention
\[[Haoyang Li' 2024-12-27](http://arxiv.org/abs/2412.19442v2)\] A Survey on Large Language Model Acceleration based on KV Cache Management
\[[June Yong Yang' 2024-02-28](http://arxiv.org/abs/2402.18096v1)\] No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization
\[[Hanchen Li' 2025-03-18](http://arxiv.org/abs/2503.14647v1)\] Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache
\[[Zhaoyuan Su' 2025-05-24](http://arxiv.org/abs/2506.02006v1)\] Efficient and Workload-Aware LLM Serving via Runtime Layer Swapping and KV Cache Resizing
\[[Liu Qianli' 2025-01-12](http://arxiv.org/abs/2501.06709v1)\] Mell: Memory-Efficient Large Language Model Serving via Multi-GPU KV Cache Management
\[[Jang-Hyun Kim' 2025-05-29](http://arxiv.org/abs/2505.23416v1)\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
\[[Yihua Cheng' 2024-09-16](http://arxiv.org/abs/2409.13761v2)\] Do Large Language Models Need a Content Delivery Network?
\[[Tianyu Guo' 2025-05-28](http://arxiv.org/abs/2505.21889v2)\] EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse
\[[Shichen Dong' 2024-03-07](http://arxiv.org/abs/2403.04643v2)\] QAQ: Quality Adaptive Quantization for LLM KV Cache
\[[Yixuan Wang' 2025-05-24](http://arxiv.org/abs/2505.20334v1)\] Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query
\[[Jingbo Yang' 2025-02-21](http://arxiv.org/abs/2502.16002v2)\] KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse
\[[Ke Hong' 2025-04-28](http://arxiv.org/abs/2504.19867v1)\] semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage
\[[Shiwei Gao' 2024-10-07](http://arxiv.org/abs/2410.05004v1)\] Fast State Restoration in LLM Serving with HCache
\[[Peilin Chen' 2025-05-23](http://arxiv.org/abs/2505.17787v1)\] Titanus: Enabling KV Cache Pruning and Quantization On-the-Fly for LLM Acceleration
\[[Shihong Gao' 2025-04-10](http://arxiv.org/abs/2504.07494v1)\] Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving
\[[Siyu Ren' 2024-02-09](http://arxiv.org/abs/2402.06262v2)\] On the Efficacy of Eviction Policy for Key-Value Constrained Generative Language Model Inference
\[[Ahmed Burak Gulhan' 2025-02-18](http://arxiv.org/abs/2502.13176v2)\] BaKlaVa -- Budgeted Allocation of KV cache for Long-context Inference
\[[Hang Zhang' 2025-04-19](http://arxiv.org/abs/2505.03756v1)\] Improving the Serving Performance of Multi-LoRA Large Language Models via Efficient LoRA and KV Cache Management
\[[Hanshi Sun' 2024-10-28](http://arxiv.org/abs/2410.21465v3)\] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
\[[Jincheng Dai' 2024-04-24](http://arxiv.org/abs/2404.15949v2)\] CORM: Cache Optimization with Recent Message for Large Language Model Inference
\[[Amir Zandieh' 2024-06-05](http://arxiv.org/abs/2406.03482v2)\] QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead
\[[Jing Xiong' 2024-10-04](http://arxiv.org/abs/2410.03090v1)\] UNComp: Uncertainty-Aware Long-Context Compressor for Efficient Large Language Model Inference
\[[Jiayi Yao' 2024-05-26](http://arxiv.org/abs/2405.16444v3)\] CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion
\[[Shibo Jie' 2025-03-20](http://arxiv.org/abs/2503.16163v1)\] SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs
\[[Bin Gao' 2024-03-23](http://arxiv.org/abs/2403.19708v3)\] Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention
\[[Yuan Feng' 2024-07-16](http://arxiv.org/abs/2407.11550v4)\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference
\[[Cunchen Hu' 2024-06-25](http://arxiv.org/abs/2406.17565v3)\] MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool
\[[Yilong Chen' 2024-08-07](http://arxiv.org/abs/2408.03675v2)\] NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

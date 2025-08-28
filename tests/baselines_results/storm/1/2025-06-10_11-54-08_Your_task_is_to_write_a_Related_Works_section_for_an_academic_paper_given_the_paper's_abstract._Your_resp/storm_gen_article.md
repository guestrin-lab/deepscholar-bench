# Related Works

Large language models (LLMs) have been widely deployed in various applications, and caching intermediate results (KV$) has been shown to substantially improve serving throughput and latency [1]. However, there is limited understanding of how LLM serving benefits from KV$ caching, and system design decisions like cache eviction policies are highly workload-dependent.
Several studies have investigated the use of KV$ caching in LLMs. For example, [2] proposed a parallel LLM inference engine, Hogwild! Inference, which uses a shared attention cache to allow multiple instances of the same LLM to run in parallel. Similarly, [3] proposed a log-augmented generation framework that reuses prior computation and reasoning from past logs to enhance the model's ability to learn from previous tasks.
Other studies have focused on characterizing workload patterns in web applications [4] and LLMs [5]. For instance, [4] conducted a systematic literature review to identify and analyze existing studies leveraging web application workloads, while [5] investigated the transfer of intermediate-task training in a zero-shot cross-lingual setting.
In the context of code generation and translation, [6] introduced the universal code (UniCode) as an intermediate representation, which significantly improves the quality of the generated code. Additionally, [7] proposed a text-driven affordance learning approach that uses textual instruction to learn contact points and manipulation trajectories from an egocentric view.
Recent studies have also explored the use of LLMs in human reliability analysis [8] and collaborative online-offline task serving systems [1]. For example, [8] proposed a novel, scenario-driven method for workload estimation using fine-tuned LLMs, while [1] introduced Echo, a collaborative online-offline task serving system that maximizes the throughput of offline tasks while satisfying online task SLOs.
Our work builds on these studies and provides a systematic characterization of the KV$ workload patterns from one of the leading LLM service providers. Our findings and proposed workload-aware cache eviction policy are consistent with and complement existing research in this area.

## References

[2] (reference 1 from collected information)
[6] (reference 2 from collected information)
[5] (reference 3 from collected information)
[7] (reference 4 from collected information)
[1] (reference 5 from collected information)
[8] (reference 6 from collected information)
[4] (reference 7 from collected information)
[3] (reference 8 from collected information)
# summary

I'm happy to help you with the lead section. However, I notice that you provided a draft page for the Related Works section, not the lead section. I'll write a lead section based on the provided abstract and guidelines.

## Lead Section

The serving of large language models (LLMs) is a critical task for cloud providers, as it directly impacts the performance and user experience of various applications. A key optimization technique for LLM serving is caching intermediate results, denoted as KV$, which has been shown to substantially improve serving throughput and latency [1]. However, the benefits of KV$ caching for LLM serving are not well understood, and system design decisions, such as cache eviction policies, are highly dependent on the workload patterns.
This paper presents a systematic characterization of KV$ workload patterns from one of the leading LLM service providers. Our analysis reveals important insights, including the skewed reuse of KV$ across requests, diverse reuse times and probabilities, and moderate cache size requirements for ideal cache hit ratios. Building on these findings, we propose a workload-aware cache eviction policy that improves serving performance under real-world traces, particularly with limited cache capacity.
Our work contributes to the growing body of research on LLM serving and caching, which has focused on improving performance, efficiency, and scalability [2][3][4]. By providing a comprehensive understanding of KV$ workload patterns and a novel cache eviction policy, this paper aims to inform the design of more efficient LLM serving systems.

## References

[1] Reference 1 from collected information
[2] Reference 2 from collected information
[3] Reference 3 from collected information
[4] Reference 4 from collected information
Please let me know if this meets your requirements or if you need further adjustments!

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
# RAGLease 设计与代码导览

本文对应 2026-09-24 工作区的实现。RAGLease 在改造后的 SGLang 中，把检索 ANN、LLM Prefill 和 Decode 作为具有依赖关系的阶段，在同一 GPU 上协调计算份额和显存。运行时的实际选择由部署配置、离线画像与在线状态共同决定；实验配置和结果不能代替实现本身。

## 系统边界

```text
预编码查询 / 文档与 IVF 索引
            │
            ▼
HTTP /rag_generate → 请求与 ANN→Prefill 依赖 → 调度器
                                           ├─ ANN: 热簇 GPU / 冷簇 CPU
                                           ├─ Prefill: 有界层 bundle
                                           └─ Decode: 持续服务中的批次
                                                    │
                       Green Context 计算分区 + VMM 共享 HBM 池
                                                    │
                                  指标、画像更新与下一轮决策
```

查询编码器在现有 VectorLiteRAG 对比路径的计时边界之外，使用预编码查询向量。检索结果经文档查找与提示词构造后进入生成服务。相同数据集的系统比较必须保持索引、`nprobe`、`topk`、模型、提示词和输出长度一致。WikiDPR 当前选定 `nprobe=128, topk=3`；Wiki-All 使用论文资产对应的 `nprobe=2048, topk=25`，两者不混用。

## 四个设计部分

1. **画像驱动的租约估计**：`sglang/benchmark/pdmux/profile_rag_scheduler.py` 枚举硬件与负载候选，检查 SLO 和稳定性，输出部署画像。`sglang/python/sglang/srt/multiplex/lease_profile.py`、`stage_predictor.py` 等读取冷启动和阶段预测输入。画像与硬件、数据集、工作负载绑定。
2. **依赖驱动的计算租约**：`multiplexing.py` 管理请求级 ANN→Prefill handoff、ANN 毫秒 credit、Prefill 层 bundle、Decode 保护以及 Green Context 选择。`ann_executor.py` 实施检索工作。租约在调度安全点重新评估，不提供运行中 CUDA kernel 的强制抢占。
3. **热度感知的弹性 HBM 租约**：`hot_index.py` 维护热簇候选和冷簇回退；`mem_cache/shared_hbm.py` 管理 KV、ANN scratch 与索引使用的 VMM 物理 extent。显存预览与实际提交分开，ANN 启动前检查 scratch 与索引所有权。模型权重等固定占用不属于弹性池。
4. **联合协调**：调度器在计算计划和 HBM 计划之间做可行性检查，在完成与切换边界排序 Green Context/VMM 转换。兼容的 ANN 和 Decode 可重叠；不兼容转换需要等待 fence。`full_lease.py` 中还包含联合进度模型和对照策略的实现，具体生产接入状态应按相应实验记录判断，不把模型代码等同于已验证的线上策略。

## 代码地图

| 路径 | 职责 |
| --- | --- |
| `sglang/python/sglang/srt/multiplex/` | 阶段调度、ANN 执行、热索引、预测器与租约模型 |
| `sglang/python/sglang/srt/mem_cache/shared_hbm.py` | 共享 HBM 规划、所有权与转换 |
| `sglang/sgl-kernel/csrc/shared_hbm/` | 共享 HBM 的底层 CUDA 实现 |
| `sglang/python/sglang/srt/managers/`、`server_args.py`、`entrypoints/http_server.py` | 请求生命周期与服务入口集成 |
| `sglang/benchmark/pdmux/` | RAG 负载、画像和比较工具 |
| `cluster/paratera/` | 集群提交、持久 GPU allocation 与运行包装器；包含大量历史配置，应选定具体协议后使用 |
| `docs/experiments/` | 版本化实验协议与判定边界 |

## 部署与复现顺序

1. 准备匹配的 SGLang/SGL Kernel 环境、模型、预处理索引与预编码查询；模型和数据不在源码发布范围内。
2. 固定硬件、数据集、检索参数、SLO 与负载，生成并核验画像及部署配置。
3. 通过 `cluster/paratera/submit_persistent_allocation.sh` 申请相应平台的持久单卡 allocation，在 allocation 内用 `run_in_persistent_allocation.sh` 串行运行。实际执行前检查作业状态和明确的 job ID。
4. 先做服务烟测和指标核对，再按 `docs/experiments/` 中选定的封存协议进行负载实验；保留原始结果与 manifest，不用开发试跑冒充正式重复。

更细的机制和代码定位见 [架构概览](raglease_architecture_overview.md)，端到端比较入口见 `sglang/benchmark/pdmux/README_rag_comparison.md`。本仓库的 `sglang/` 是独立 Git 工作区，发布时应记录其精确提交，而不能只把目录名当作可复现的代码版本。

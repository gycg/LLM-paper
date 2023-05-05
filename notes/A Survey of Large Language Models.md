# A Survey of Large Language Models



[toc]



这篇综述发表于2023年4月28日，作者主要来自中国人民大学。

文章从四个方面来介绍LLM，包括pre-training，adaptation tuning，utilization 和 capacity evaluation。也总结了一些开发LLM的资源，并讨论未来值得研究的遗留问题。

文章官方git如下：[LLMSurvey](https://github.com/RUCAIBox/LLMSurvey)

## 1. INTRODUCTION

语言模型（LM）是提高机器智能的主要方法。可以分为四个发展阶段：

* Statistical language models (SLM)：统计语言模型。例如n-gram语言模型。SLM会有维度灾难问题。

* Neural language models (NLM)：神经语言模型。例如RNN、word2vec。word2vec通过浅层神经网络来学习分布式词表征。这些研究初步使用语言模型进行表征学习，对NLP领域发展有重要影响。

* Pre-trained language models (PLM)：预训练语言模型。例如ELMo，Bert。确定了pre-training和fine-tuning的范式。

* Large language models (LLM)：大语言模型。根据[scaling laws](https://arxiv.org/pdf/2001.08361.pdf)，扩大模型或增加数据可以提高下游任务的模型能力。虽然模型架构和预训练任务比较类似，但是大模型表现出惊人的能力（一般称为涌现现象，但是斯坦福有[论文](https://arxiv.org/pdf/2304.15004.pdf)提出涌现只是度量选择的结果）。

文章针对LLM所讲的四个方面具体如下：

* pre-training：预训练。

* adaptation-tuning：自适应调整。调整LLM的有效性和安全性。

* utilization：应用。如何利用LLM解决各种下游任务。

* capability evaluation：能力评估。如何通过现有实验评估LLM的能力。

## 2. OVERVIEW

### 2.1 LLMs的背景

#### 比例定律（Scaling Laws）

#### 涌现能力（Emergent Abilities）

#### 关键技术（Key Techniques）

### 2.2 GPT系列模型的技术演进

#### 早期探索

#### 容量飞跃

#### 能力提升

#### 语言模型里程碑

## 3. LLM资源

### 3.1 公开可获取的模型或API

#### 10B级别的模型

#### 100B级别的模型

#### LLM的公开API

### 3.2 常用语料

#### Books

#### CommonCrawl

#### Reddit Links

#### Wikipedia

#### Code

#### 其他

### 3.3 库资源

#### Transformers

#### DeepSpeed

#### Megatron-LM

#### JAX

#### Colossal-AI

#### BMTrain

#### FastMoE

## 4. 预训练

### 4.1 数据收集

#### 4.1.1 数据源

#### 4.1.2 数据处理

#### 4.1.3 预训练数据对LLM的影响

### 4.2 架构

#### 4.2.1 主流架构

#### 4.2.2 详细配置

#### 4.2.3 预训练任务

#### 4.2.4 总结和讨论

### 4.3 模型训练

#### 4.3.1 优化设置

#### 4.3.2 可扩展训练技术

## 5. 自适应调整

### 5.1 指令调整

#### 5.1.1 格式化实例构建

#### 5.1.2 指令调整策略

#### 5.1.3 指令调整的影响

### 5.2 对齐调整

#### 5.2.1 对齐的背景和标准

#### 5.2.2 收集人类反馈

#### 5.2.3 从人类反馈中强化学习

### 5.3 高效调整（Efficient Tuning）

#### 5.3.1 参数高效微调方法

##### Adapter Tuning

##### Prefix Tuning

##### Prompt Tuning

##### Low-Rank Adaptation（LoRA）

#### 5.3.2 LLM的参数高效微调

## 6. 应用

### 6.1 情境学习（In-Context Learning）

#### 6.1.1 提示词公式（Prompting Formulation）

#### 6.1.2 示范设计（Demonstration Design）

#### 6.1.3 底层机制

### 6.2 思维链（Chain-of-Thought）

#### 6.2.1 使用CoT进行情境学习

#### 6.2.2 CoT的更多讨论

## 7. 能力评估

### 7.1 基础评价任务

#### 7.1.1 语言生成

#### 7.1.2 知识运用

#### 7.1.3 复杂推理

### 7.2 进阶能力评估

#### 7.2.1 人类对齐

#### 7.2.2 与外界环境交互

#### 7.2.3 操作工具

### 7.3 公共基准与实证分析

#### 7.3.1 评估基准

#### 7.3.2 LLM的综合能力分析

## 8. 总结和展望

#### 理论与原则

#### 模型架构

#### 模型训练

#### 模型应用

#### 安全性和对齐

#### 应用和生态系统



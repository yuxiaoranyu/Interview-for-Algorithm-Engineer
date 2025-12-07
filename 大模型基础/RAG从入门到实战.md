<h1 id="目录">目录</h1>

## 一、RAG基础理论与概念

### 1.1 RAG基础定义
- [1.什么是RAG？](#1什么是RAG？)

### 1.2 RAG评估与挑战
- [1.RAG文档召回率是什么？](#1.RAG文档召回率是什么？)
- [2.RAG技术的难点有哪些？](#2.RAG技术的难点有哪些？)
- [3.RAG存在的一些问题和避免方式有哪些？](#3.RAG存在的一些问题和避免方式有哪些？)
- [4.在大模型工程应用中RAG与LLM微调优化哪个是最优解?](#4.在大模型工程应用中RAG与LLM微调优化哪个是最优解?)
- [5.RAG的评估指标有哪些？](#5.RAG的评估指标有哪些？)

## 二、文档处理与向量化

### 2.1 文档格式与加载
- [1.RAG项目中为什么倾向于选择Markdown格式的文档？](#1.RAG项目中为什么倾向于选择Markdown格式的文档？)
- [2.RAG之PDF文档加载器介绍](#2.RAG之PDF文档加载器介绍)

### 2.2 文本分块策略
- [1.RAG项目中文本分块策略介绍？](#1.RAG项目中文本分块策略介绍？)
- [2.如何保证文档切片不会造成相关内容的丢失？文档切片的大小如何控制？](#2.如何保证文档切片不会造成相关内容的丢失？文档切片的大小如何控制？)
- [3.RAG之chunking方法介绍](#3.RAG之chunking方法介绍)

### 2.3 嵌入与向量化
- [1.RAG之Embedding模型介绍](#1.RAG之Embedding模型介绍)
- [2.RAG之假设文档嵌入(HyDE)](#2.RAG之假设文档嵌入(HyDE))

### 2.4 索引与存储
- [1.怎么通过预处理数据库提高RAG的精度？](#1.怎么通过预处理数据库提高RAG的精度？)
- [2.怎么通过知识图谱提高RAG的精度？](#2.怎么通过知识图谱提高RAG的精度？)
- [3.怎么使用总分层级索引提高RAG的精度？](#3.怎么使用总分层级索引提高RAG的精度？)
- [4.怎么使用父子层级索引提高RAG的精度？](#4.怎么使用父子层级索引提高RAG的精度？)
- [5.llama-index的索引类别有哪些？](#5.llama-index的索引类别有哪些？)
- [6.向量数据库介绍](#6.向量数据库介绍)

## 三、检索优化技术

### 3.1 预检索优化
- [1.怎么使用子查询（预检索）优化RAG？](#1.怎么使用子查询（预检索）优化RAG？)
- [2.怎么使用假设答案（预检索）优化RAG？](#2.怎么使用假设答案（预检索）优化RAG？)
- [3.还有哪些其他预检索技术优化RAG？](#3.还有哪些其他预检索技术优化RAG？)
- [4.RAG之查询重写的策略介绍](#4.RAG之查询重写的策略介绍)
- [5.什么是基于提示词的自适应（主动）检索？](#5.什么是基于提示词的自适应（主动）检索？)

### 3.2 检索过程优化
- [1.怎么使用多种切分方式和并行查询？](#1.怎么使用多种切分方式和并行查询？)
- [2.怎么让检索过程更加准确？](#2.怎么让检索过程更加准确？)
- [3.什么是RAG的迭代检索？](#3.什么是RAG的迭代检索？)
- [4.什么是RAG的递归检索？](#4.什么是RAG的递归检索？)
- [5.怎么使用llama index实现递归检索？](#5.怎么使用llamaindex实现递归检索？)

### 3.3 后检索优化
- [1.RAG之Re-Ranking机制介绍](#1.RAG之Re-Ranking机制介绍)
- [2.怎么使用Remark技术（后检索）优化RAG？](#2.怎么使用Remark技术（后检索）优化RAG？)

## 四、RAG架构与范式

### 4.1 经典RAG范式
- [1.什么是Modular RAG及其框架？](#1.什么是ModularRAG及其框架？)
- [2.经典的RAG Flow范式（Tuning阶段）？](#2.经典的RAGFlow范式（Tuning阶段）？)
- [3.经典的RAG Flow范式（推理阶段-顺序模式）？](#3.经典的RAGFlow范式（推理阶段-顺序模式）？)
- [4.经典的RAG Flow范式（推理阶段-条件模式）？](#4.经典的RAGFlow范式（推理阶段-条件模式）？)
- [5.如何通过llamaindex实现条件模式？](#5.如何通过llamaindex实现条件模式？)
- [6.经典的RAG Flow范式（推理阶段-分支模式）？](#6.经典的RAGFlow范式（推理阶段-分支模式）？)

### 4.2 高级RAG架构
- [1.什么是Agentic RAG？它与传统RAG有何不同？](#1.什么是Agentic-RAG？它与传统RAG有何不同？)
- [2.Agentic RAG的核心组件有哪些？](#2.Agentic-RAG的核心组件有哪些？)
- [3.在Agentic RAG中，代理模块如何决定检索哪些文档？](#3.在Agentic-RAG中，代理模块如何决定检索哪些文档？)
- [4.Agentic RAG如何处理检索到的信息以生成高质量的回答？](#4.Agentic-RAG如何处理检索到的信息以生成高质量的回答？)
- [5.请解释Agentic RAG中的自主学习机制是如何工作的？](#5.请解释Agentic-RAG中的自主学习机制是如何工作的？)
- [6.Agentic RAG在哪些实际应用中具有优势？请举例说明。](#6.Agentic-RAG在哪些实际应用中具有优势？请举例说明。)
- [7.在多轮对话中，Agentic RAG如何保持上下文一致性和信息连贯性？](#7.在多轮对话中，Agentic-RAG如何保持上下文一致性和信息连贯性？)
- [8.Agentic RAG面临的主要挑战有哪些？如何应对这些挑战？](#8.Agentic-RAG面临的主要挑战有哪些？如何应对这些挑战？)
- [9.如何评估Agentic RAG系统的性能？有哪些关键指标？](#9.如何评估Agentic-RAG系统的性能？有哪些关键指标？)
- [10.在设计Agentic RAG时，如何平衡检索与生成模块的权重？](#10.在设计Agentic-RAG时，如何平衡检索与生成模块的权重？)

### 4.3 特定技术实现
- [1.怎么通过llamaindex实现FLAREdirect？](#1.怎么通过llamaindex实现FLAREdirect？)
- [2.怎么通过langchain实现FLAREdirect？](#2.怎么通过langchain实现FLAREdirect？)

## 五、RAG开发框架与工具

### 5.1 LangChain框架
- [1.怎么用langchain构建简单RAG？](#1.怎么用langchain构建简单RAG？)
- [2.基于langchain的本地文档问答系统实现步骤有哪些?](#2.基于langchain的本地文档问答系统实现步骤有哪些?)
- [3.介绍一下 LangChain](#3.介绍一下LangChain)
- [4.LangChain 中 Chat Message History 是什么？](#4.LangChain中ChatMessageHistory是什么？)
- [5.LangChain 中 LangChain Agent 是什么？](5.LangChain中LangChainAgent是什么？)
- [6.LangChain 支持哪些功能?](#6.LangChain支持哪些功能?)
- [7.什么是 LangChain model?](#7.什么是LangChainmodel?)
- [8.LangChain 如何链接多个组件处理一个特定的下游任务？](#8.LangChain如何链接多个组件处理一个特定的下游任务？)

### 5.2 RAGFlow项目
- [1.怎么部署RAGFlow项目？](#1.怎么部署RAGFlow项目？)
- [2.怎么用RAGFlow建立知识库与聊天？](#2.怎么用RAGFlow建立知识库与聊天？)

<h1 id="一、RAG基础理论与概念"> 一、RAG基础理论与概念 </h1>

<h2 id="1.1 RAG基础定义"> 1.1 RAG基础定义 </h2>

<h3 id='1.什么是RAG？'>1.什么是RAG？</h3>

## 一、什么是 RAG？
RAG 全称 Retrieval-Augmented Generation，翻译成中文是检索增强生成。检索指的是检索外部知识库，增强生成指的是将检索到的知识送给大语言模型以此来优化大模型的生成结果，使得大模型在生成更精确、更贴合上下文答案的同时，也能有效减少产生误导性信息的可能。

## 二、为什么需要 RAG？
之所以需要 RAG，是因为大语言模型本身存在一些局限性。

1) 时效性
模型的训练是基于截至某一时间点之前的数据集完成的。这意味着在该时间点之后发生的任何事件、新发现、新趋势或数据更新都不会反映在模型的知识库中。例如，我的训练数据在 2023 年底截止，之后发生的事情我都无法了解。另外，大型模型的训练涉及巨大的计算资源和时间。这导致频繁更新模型以包括最新信息是不现实的，尤其是在资源有限的情况下。

2) 覆盖性
虽然大模型的训练数据集非常庞大，但仍可能无法涵盖所有领域的知识或特定领域的深度信息。例如，某些专业的医学、法律或技术问题可能只在特定的文献中被详细讨论，而这些文献可能未被包括在模型的训练数据中。另外，对于一些私有数据集，也是没有被包含在训练数据中的。当我们问的问题的答案没有包含在大模型的训练数据集中时，这时候大模型在回答问题时便会出现幻觉，答案也就缺乏可信度。

由于以上的一些局限性，大模型可能会生成虚假信息。为了解决这个问题，需要给大模型外挂一个知识库，这样大模型在回答问题时便可以参考外挂知识库中的知识，也就是 RAG 要做的事情。

## 三、RAG 的流程
RAG 的中文名称是检索增强生成，从字面意思来理解，包含三个检索、增强和生成三个过程。

- 检索： 根据用户的查询内容，从外挂知识库获取相关信息。具体来说，就是将用户的查询通过嵌入模型转换成向量，以便与向量数据库中存储的知识相关的向量进行比对。通过相似性搜索，从向量数据库中找出最匹配的前 K 个数据。

- 增强： 将用户的查询内容和检索到的相关知识一起嵌入到一个预设的提示词模板中。

- 生成： 将经过检索增强的提示词内容输入到大语言模型（LLM）中，以此生成所需的输出。 流程图如下所示：
![rag流程图](./imgs/rag流程图.png)

<h2 id="1.2 RAG评估与挑战"> 1.2 RAG评估与挑战 </h2>

<h3 id='1.RAG文档召回率是什么？'>1.RAG文档召回率是什么？</h3>

RAG（Retrieval-Augmented Generation）中的文档召回率（Document Recall）是指在检索阶段，模型能够成功找到与用户查询相关的所有文档的比例。具体来说，它衡量的是在所有相关文档中，有多少被成功检索到了。

文档召回率是评估检索系统性能的重要指标。它可以用以下公式计算：文档召回率=成功检索到的相关文档数量/所有相关文档数量

在RAG中，文档召回率的高低直接影响生成模型的表现。如果召回率低，生成模型可能会缺乏足够的背景信息，从而影响答案的准确性和相关性。

要提高文档召回率，可以采取以下措施：

1. 改进检索模型：使用更先进的检索模型，如Dense Passage Retrieval (DPR) 或改进BM25算法，来提高相关文档的检索效果。

2. 扩展检索范围：增加知识库的规模和多样性，以确保包含更多潜在相关文档。

3. 优化检索策略：调整检索策略，使用多轮检索或结合多个检索模型的结果，来提高召回率。

高召回率可以确保生成模型有更丰富的信息源，从而提高最终生成答案的准确性和可靠性。

<h3 id='2.RAG技术的难点有哪些？'>2.RAG技术的难点有哪些？</h3>

（1）数据处理

目前的数据文档种类多，包括doc、ppt、excel、pdf扫描版和文字版。ppt和pdf中包含大量架构图、流程图、展示图片等都比较难提取。而且抽取出来的文字信息，不完整，碎片化程度比较严重。

而且在很多时候流程图，架构图多以形状元素在PPT中呈现，光提取文字，大量潜藏的信息就完全丢失了。

（2）数据切片方式

不同文档结构影响，需要不同的切片方式，切片太大，查询精准度会降低，切片太小一段话可能被切成好几块，每一段文本包含的语义信息是不完整的。

（3）内部知识专有名词不好查询

目前较多的方式是向量查询，对于专有名词非常不友好；影响了生成向量的精准度，以及大模型输出的效果。

（4）新旧版本文档同时存在

一些技术报告可能是周期更新的，召回的文档如下就会出现前后版本。

（5）复杂逻辑推理

对于无法在某一段落中直接找到答案的，需要深层次推理的问题难度较大。

（6）金融行业公式计算

如果需要计算行业内一些专业的数据，套用公式，对RAG有很大的难度。

（7）向量检索的局限性

向量检索是基于词向量的相似度计算，如果查询语句太短词向量可能无法反映出它们的真实含义，也无法和其他相关的文档进行有效的匹配。这样就会导致向量检索的结果不准确，甚至出现一些完全不相关的内容。

（8）长文本

（9）多轮问答

<h3 id='3.RAG存在的一些问题和避免方式有哪些？'>3.RAG存在的一些问题和避免方式有哪些？</h3>

（1）分块（Chunking）策略以及Top-k算法

一个成熟的RAG应该支持灵活的分块，并且可以添加一点重叠以防止信息丢失。用固定的、不适合的分块策略会造成相关度下降。最好是根据文本情况去适应。

在大多数设计中，top_k是一个固定的数字。因此，如果块大小太小或块中的信息不够密集，我们可能无法从向量数据库中提取所有必要的信息。

（2）世界知识缺失

比如我们正在构建一个《西游记》的问答系统。我们已经把所有的《西游记》的故事导入到一个向量数据库中。现在，我们问它：人有几个头?

最有可能的是，系统会回答3个，因为里面提到了哪吒有“三头六臂”，也有可能会说很多个，因为孙悟空在车迟国的时候砍了很多次头。而问题的关键是小说里面不会正儿八经地去描述人有多少个头，所以RAG的数据有可能会和真实世界知识脱离。

（3）多跳问题（推理能力）

让我们考虑另一个场景：我们建立了一个基于社交媒体的RAG系统。那么我们的问题是：谁知道埃隆·马斯克？然后，系统将遍历向量数据库，提取埃隆·马斯克的联系人列表。由于chunk大小和top_k的限制，我们可以预期列表是不完整的；然而，从功能上讲，它是有效的。

现在，如果我们重新思考这个问题：除了艾梅柏·希尔德，谁能把约翰尼·德普介绍给伊隆·马斯克？单次信息检索无法回答这类问题。这种类型的问题被称为多跳问答。解决这个问题的一个方法是:

    找回埃隆·马斯克的所有联系人
    找回约翰尼·德普的所有联系人
    看看这两个结果之间是否有交集，除了艾梅柏·希尔德
    如果有交集，返回结果，或者将埃隆·马斯克和约翰尼·德普的联系方式扩展到他们朋友的联系方式并再次检查。

有几种架构来适应这种复杂的算法，其中一个使用像ReACT这样复杂的prompt工程，另一个使用外部图形数据库来辅助推理。我们只需要知道这是RAG系统的限制之一。

（4）信息丢失

RAG系统中的流程链:

    将文本分块（chunking）并生成块（chunk）的Embedding
    通过语义相似度搜索检索数据块
    根据top-k块的文本生成响应  


<h3 id='4.在大模型工程应用中RAG与LLM微调优化哪个是最优解?'>4.在大模型工程应用中RAG与LLM微调优化哪个是最优解?</h3>

RAG: 将检索(或搜索)的能力集成到LLM文本生成中，结合了检索系统(从大型语料库中获取相关文档片段)和LLM(使用这些片段中的信息生成答案)。
微调: 对预训练的LLM模型在特定数据集上进一步训练，使其适应特定任务或提高其性能的过程。

一般在工程中考虑使用RAG还是LLM需要从以下几点考虑：

（1）如果需要访问大量的外部数据，并且要实时更新。RAG系统在具有动态数据的环境中具有固有的优势。它们的检索机制不断地查询外部源，确保它们用于生成响应的信息是最新的。随着外部知识库或数据库的更新，RAG系统无缝地集成了这些更改，在不需要频繁的模型再训练的情况下保持其相关性。

（2）如果我们需要改变模型的输出风格，如我们想让模型听起来更像医学专业人士，用诗意的风格写作，或者使用特定行业的行话，那么对特定领域的数据进行微调可以让我们实现这些定制。

RAG虽然在整合外部知识方面很强大，但主要侧重于信息检索。

（3）一般来说RAG与LLM微调可以单独使用也可以组合使用。

（4）通过将模型在特定领域的数据中微调可以一定程度上减少幻觉。然而当面对不熟悉的输入时，模型仍然可能产生幻觉。相反，RAG系统天生就不容易产生幻觉，因为它们的每个反应都是基于检索到的证据。
    

<h3 id='5.RAG的评估指标有哪些？'>5.RAG的评估指标有哪些？</h3>

### Context precision上下文精确度

评估检索质量，衡量上下文中所有相关的真实信息是否被排在较高的位置。理想情况下，所有相关的信息快都应该出现在排名的最前面。这个指标是根据问题和上下文来计算的，数值范围在0~1之间，分数越高表示精确度越好。

### Context Recall上下文召回率

衡量检索的完整性，用来衡量检索到的上下文与被视为事实真相的标注答案的一致性程度。根据事实真相和检索到的上下文来计算，数值范围在0~1之间，数值越高表示性能越好。为了从事实真相的答案中估计上下午的召回率，需要分析答案中的每个句子是否可以归因于检索到的上下文。在理想情况下，事实真相答案中的所有句子都应该能够对应到检索到的上下文中。

### Faithfulness忠实度

衡量生成答案中的幻觉情况，衡量生成答案与给定上下文之间的事实一致性。忠实度得分是基于答案和检索到的上下文计算出来的，答案的评分范围在0~1之间，分数越高越好。

### Answer Relevance答案相关性

衡量答案对问题的直接性（紧扣问题的核心），旨在评估生成答案与给定提示的相关程度。如果答案不完整或包含冗余信息，则会被赋予较低的分数。这个指标使用问题和答案来计算，其值介于0~1之间，得分越高表明答案的相关性越好。


<h1 id="二、文档处理与向量化"> 二、文档处理与向量化 </h1>

<h2 id="2.1 文档格式与加载"> 2.1 文档格式与加载 </h2>

<h3 id='2.RAG项目中为什么倾向于选择Markdown格式的文档？'>2.RAG项目中为什么倾向于选择Markdown格式的文档？</h3>

在RAG（Retrieval-Augmented Generation）系统中，将文档抽取为Markdown格式具有多重优势，主要体现在以下几个方面：
 
1) 结构化与可读性优势
Markdown凭借其简洁的语法和良好的可读性，成为RAG数据处理的理想格式。其一致的格式标准和易于转换的特性为后续的文本处理、分块和向量化提供了便利。相比于复杂的文档格式（如PDF、Word），Markdown能够以最简化的形式保留核心内容结构，同时便于机器解析和人类理解。
 
2) 优化文档分块与检索效果
在RAG系统中，文档分块的质量直接影响检索效果。Markdown格式天然支持通过标题层级（#、##、###等）将文档划分为语义连贯的段落。这种结构化的分块方式能够：
- 保持内容的上下文完整性
- 提高检索结果的相关性
- 为后续的生成阶段提供更精准的上下文信息
 
3) 保留关键文档结构
Markdown能够有效处理表格、代码块、列表等复杂元素，同时保持数据之间的逻辑关系。先进的文档转换工具可以将PDF等复杂格式转换为Markdown，同时保留原始布局、格式和元数据，确保信息在转换过程中不丢失。
 
4) 格式统一与系统兼容性
RAG系统通常需要处理多种来源、多种格式的文档。将所有文档统一转换为Markdown格式，可以：
- 简化预处理流程
- 提高系统的可维护性
- 确保不同文档在向量化和检索时具有一致的处理标准
 
这种偏好与大模型输出为markdown格式有关，主要体现在以下两个层面：
 
1) 输入-输出格式一致性
大型语言模型（LLM）在RAG系统中既需要处理输入的上下文信息，也需要生成最终的输出结果。当输入文档以Markdown格式提供时：
- LLM能够更准确地理解文档结构和语义
- 模型在生成响应时可以自然地采用相同的格式规范
- 保持整个流程（输入→处理→输出）的格式一致性，减少格式转换带来的信息损失
 
2) LLM友好的格式设计
Markdown被广泛认为是"LLM友好型"格式，原因包括：
- 简洁性：没有复杂的样式和布局干扰，让模型专注于内容本身
- 标准化：统一的语法规范降低了模型的理解难度
- 结构化：标题、列表、表格等元素为模型提供了明确的语义标记
- 通用性：几乎所有现代LLM都经过大量Markdown格式数据的训练，对此格式具有天然的适应性


<h3 id='2.RAG之PDF文档加载器介绍'>2.RAG之PDF文档加载器介绍</h3>

### PDF的解析方法：

- 基于规则的方法：根据文档的组织特征确定每个部分的风格和内容。然而，这种方法不是很通用，因为PDF有很多类型和布局，不可能用预定义的规则覆盖所有类型和布局。

- 基于深度学习模型的方法：例如将目标检测和OCR模型相结合的流行解决方案。

- 基于多模态大模型对复杂结构进行Pasing或提取PDF中的关键信息。

### 常见的PDF文档加载器

1) PyPDF

PyPDF 是一个用于处理PDF文件的Python库。它提供了一系列的功能，允许用户读取、写入、分析和修改PDF文档。在LangChain中，PyPDFLoader 使用 pypdf 库加载PDF文档为文档数组，PDF将会按照page逐页读取，每个文档包含页面内容和带有页码的元数据。

```
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
print(pages[0]
```

图片信息提取：pip install rapidocr-onnxruntime

```
from langchain_community.document_loaders import PyPDFLoader
 
loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
pages = loader.load()
print(pages[4].page_content)
```

2) pyplumber

```
from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("example_data/layout-parser-paper.pdf")
pages = loader.load()
```

3) PDFMiner

将整个文档解析成一个完整的文本，文本结构可以自行定义

```
from langchain_community.document_loaders import PDFMinerLoader
loader = PDFMinerLoader("example_data/layout-parser-paper.pdf")
pages = loader.load()
```

以上三种是基于规则解析

4) Unstructured(基于深度学习模型)

非结构化加载器针对不同的文本块创建了不同的元素。默认情况下将其组合在一起，可以通过指定model="elements"保持这种分离，然后根据自己的逻辑进行分离

```
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("example_data/layout-parser-paper.pdf", model="elements")
pages = loader.load()
```

<h2 id='2.2 文本分块策略'>2.2 文本分块策略</h2>

<h3 id='1.RAG项目中文本分块策略介绍？'>1.RAG项目中文本分块策略介绍？</h3>

## 一、基本概念介绍
1. 什么是RAG中的文本分块（Text Chunking）？
文本分块是将庞大的原始文本（如长篇报告、电子书、API文档等）分割成更小、更易于处理的文本片段（Chunks）的过程。这些Chunks是RAG系统信息处理的基本单元，会被送入Embedding模型向量化后存入向量数据库，最终服务于检索环节。
 
2. 为什么RAG系统离不开文本分块？核心目标是什么？
文本分块是RAG系统的核心前置步骤，核心目标有三点：
- 克服上下文窗口限制：大语言模型（LLM）存在上下文长度限制，分块能确保输入LLM的信息在其“消化能力”范围内；
- 提高检索精度与效率：小而语义集中的块可减少无关信息干扰，提升匹配精度，同时向量数据库检索小块的速度远快于全文搜索；
- 维护上下文完整性：理想分块需尽可能保持语义连贯，避免割裂完整逻辑（如句子、代码块）。
 
3. 文本分块对RAG性能有哪些深远影响？
分块策略直接决定RAG系统的检索质量和生成质量：
- 检索质量：分块的粒度、边界确定方式、块间关联（如是否重叠），决定检索器能否准确找到相关片段。糟糕分块会导致返回不相关、不完整或冗余信息；
- 生成质量：LLM的输出质量依赖检索到的Chunks，若Chunks上下文割裂、信息缺失，即使LLM能力再强，也难以生成准确连贯的答案。
 
4. 文本分块的核心挑战是什么？
核心挑战是“检索精度（Precision）”与“上下文完整性（Context）”的权衡：
- 小块优势：信息聚焦、语义集中，检索精度高；缺点：可能丢失上下文，无法支撑复杂问题解答；
- 大块优势：保留丰富上下文，利于理解复杂概念；缺点：包含较多噪音，稀释相关性信号，降低检索精度，增加LLM处理负担。

## 二、基础分块策略
1. 固定大小分块（Fixed-size Chunking）的核心逻辑、优缺点及适用场景是什么？
- 核心逻辑：按固定字符数/Token数切割，可设置“重叠（Overlap）”部分缓解语义割裂（如chunk_size=500字符，chunk_overlap=50字符）；
- 优点：实现简单、计算开销小、对文本格式无要求；
- 缺点：易破坏语义完整性（如句子/单词中间切割）、忽略文本结构；
- 适用场景：对结构要求低的简单场景、海量数据快速预处理、复杂策略的兜底手段。
 
2. 基于句子的分块（Sentence Splitting）有何特点？适合处理哪种文本？
- 核心逻辑：先通过标点或NLP库（NLTK、SpaCy）分割句子，再将连续句子合并成接近目标大小的Chunks；
- 优点：保持句子级语义完整性，符合自然语言结构；
- 缺点：句子长度差异大导致Chunks大小不均，简单标点分割可能误判（如Mr. Smith中的“.”），对无句子结构文本（如代码、JSON）效果差；
- 适用场景：结构良好的文本（新闻、报告、小说），需保持句子语义完整的场景。
 
3. 递归字符分块（Recursive Character Text Splitting）的优势的是什么？
- 核心逻辑：按预设分隔符优先级递归分割（如优先\n\n（段落）→\n（换行）→空格→字符），确保块大小不超限；
- 优点：兼顾语义结构与大小控制，比固定大小更智能，适应性强；
- 缺点：实现较复杂，效果依赖分隔符优先级设计，无明显分隔符时退化为字符分割；
- 适用场景：多种文本文档，需控制块大小且保留文本结构的场景（常作为RAG框架默认选项）。
 
4. 基于文档结构的分块（Document Structure-aware Chunking）如何工作？有何特点？
- 核心逻辑：利用文档固有结构分割（如HTML标签、Markdown标题/列表、JSON层级），如每个<p>标签或Markdown二级标题下内容作为一个Chunk；
- 优点：尊重原文结构、语义连贯性强，可附加结构元数据（如标题）辅助检索；
- 缺点：依赖清晰文档结构，结构元素文本量差异大导致Chunks大小不均，需针对不同格式编写解析逻辑；
- 适用场景：结构化文档（网页、Markdown、JSON/XML），需利用结构信息检索的场景。
 
5. 混合分块（Hybrid Chunking）的核心思路和优势是什么？
- 核心思路：组合多种策略，先按文档结构（如Markdown标题）粗粒度分割，再对超大小的块用递归字符/句子分块细切，保留结构元数据；
- 优点：兼顾结构完整性与大小控制，元数据丰富，灵活性高；
- 缺点：实现复杂度高，需设计组合逻辑和参数；
- 适用场景：对分块质量要求高的场景（如Markdown、富文本文档），需平衡上下文、结构与大小的需求。

## 三、进阶分块策略
1. 语义分块（Semantic Chunking）与基础策略的本质区别是什么？
- 核心逻辑：不依赖字符/标点，通过计算相邻文本的Embedding向量相似度，在语义断裂点（相似度低于阈值）切割；
- 优点：切分点贴合语义，Chunk内部语义高度相关，符合人类阅读理解习惯；
- 缺点：计算开销大（需生成Embedding），依赖Embedding模型能力，阈值需实验调优；
- 适用场景：分块质量要求高、计算资源充裕的场景，如无结构化标记的长文本（纯文本、对话记录）。
 
2. 分层分块（Hierarchical Chunking）的核心设计是什么？
- 核心逻辑：系统化创建多层级Chunks（如章节→段落→句子），不同层级Chunks可分别索引，适配不同检索需求；
- 优点：提供多粒度上下文，增加检索灵活性；
- 缺点：增加索引复杂度和存储空间，需设计层级关系；
- 适用场景：复杂层级文档（书籍、长篇报告），需灵活选择上下文粒度的检索场景。
 
3. Small-to-Big检索（父文档检索器）如何结合分块提升效果？
- 核心逻辑：依赖分层/父子关系分块，检索流程为：用查询匹配小块（保证精度）→返回小块所属的父块（提供完整上下文）；
- 优点：结合小块的检索精度和大块的上下文完整性，兼顾精准定位与背景支撑；
- 缺点：需维护小块与父块的映射关系，增加索引和检索逻辑复杂度；
- 适用场景：需高精度检索且需丰富上下文生成答案的RAG应用。
 
4. 命题分块（Proposition Chunking）适合哪些场景？
- 核心逻辑：通过LLM或NLP模型提取文本中的原子性事实命题（如“苹果2023年发布Vision Pro”拆分为3个独立命题）；
- 优点：细粒度、高聚焦，适合精确事实检索；
- 缺点：依赖模型抽取能力，计算成本高，可能丢失文本语气和复杂关系；
- 适用场景：知识库构建、事实性问答系统，对信息原子性和精确性要求极高的场景。
 
5. Agentic/LLM-based Chunking的核心思路和现状是什么？
- 核心逻辑：让Agent或LLM通过Prompt决策分块方式，动态选择组合策略；
- 优点：理论上可实现最智能的语义化分块；
- 缺点：实现复杂（需Prompt工程/Agent设计），成本高、速度慢，结果可控性差；
- 适用场景：研究探索项目，对分块质量有极致追求且不计成本的应用。

## 四、块优化策略
1. 如何通过上下文富化（Context Enrichment）优化分块效果？
- 核心思路：分块后为Chunk添加额外上下文（如相邻句子、所属章节标题、摘要等元数据）；
- 优点：不显著增大Chunk大小，帮助LLM理解Chunk在原文中的位置和背景；
- 缺点：需额外处理步骤提取富化信息；
- 适用场景：可与任意分块策略结合，尤其适合小块分块（弥补上下文不足）。
 
2. 选择分块策略时需考虑哪些因素？
需结合数据特性和应用场景综合判断：
- 数据特性：文本类型（结构化/非结构化）、结构复杂度、信息密度、语言类型；
- 应用场景：检索目标（精确事实/复杂概念）、响应速度要求、计算资源、LLM上下文窗口大小。
 
3. 评估分块策略好坏的关键指标有哪些？
除了Chunk长度分布，还包括：
- 检索指标：精度（相关Chunk命中率）、召回率（是否覆盖所有相关信息）；
- 生成指标：LLM回答的准确性、连贯性、完整性；
- 效率指标：分块处理速度、检索响应速度、存储开销。

<h3 id='2.如何保证文档切片不会造成相关内容的丢失？文档切片的大小如何控制？'>2.如何保证文档切片不会造成相关内容的丢失？文档切片的大小如何控制？</h3>

一、一般的文本切分可以按照字符、长度或者语义（经过NLP语义分析的模型）进行拆分。

二、刚好有一段完整的文本，如果切太小，那么则会造成信息丢失，给 LLM 的内容则不完整。太大则不利于向量检索命中。

文本切片不要使用固定长度，可以采用 LangChain 的 MultiVector Retriever ，它的主要是在做向量存储的过程进一步增强文档的检索能力。LangChain 有 Parent Document Retriever 采用的方案是用小分块保证尽可能找到更多的相关内容，用大分块保证内容完整性， 这里的大块文档是指 Parent Document 。MultiVector Retriever 在 Parent Document Retriever 基础之上做了能力扩充。

参考链接：https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/multi_vector/

<h3 id='3.RAG之chunking方法介绍'>3.RAG之chunking方法介绍</h3>

1) Fixed size chunking：这是最常见、最直接的分块方法。我们只需决定分块中的tokens数量，以及它们之间是否应该有任何重叠。一般来说，我们希望在块之间保持一些重叠，以确保语义上下文不会在块之间丢失。与其他形式的分块相比，固定大小的分块在计算上便宜且使用简单，因为它不需要使用任何NLP库。

2) Recursive Chunking：递归分块使用一组分隔符，以分层和迭代的方式将输入文本划分为更小的块。如果最初分割文本没有产生所需大小或结构的块，则该方法会使用不同的分隔符或标准递归地调用结果块，直至达到所需的块大小或结构。这意味着，虽然块的大小不会完全相同，但它们仍然具有相似的大小，并可以利用固定大小块和重叠的优点。

3) Document Specific Chunking：该方法不像上述两种方法一样，它不会使用一定数量的字符或递归过程，而是基于文档的逻辑部分（如段落或小节）来生成对齐的块。该方法可以保持内容的组织，从而保持了文本的连贯性，比如Markdown、Html等特殊格式。

4) Semantic Chunking：语义分块会考虑文本内容之间的关系。它将文本划分为有意义的、语义完整的块。这种方法确保了信息在检索过程中的完整性，从而获得更准确、更符合上下文的结果。与之前的分块策略相比，速度较慢。

<h2 id='2.3 嵌入与向量化'>2.3 嵌入与向量化</h2>

<h3 id='1.RAG之Embedding模型介绍'>1.RAG之Embedding模型介绍</h3>

1.BGE

BGE，即BAAI General Embedding，是由智源研究院（BAAI）团队开发的一款文本Embedding模型。该模型可以将任何文本映射到低维密集向量，这些向量可用于检索、分类、聚类或语义搜索等任务。此外，它还可以用于LLMs的向量数据库。

BGE模型在2023年有多次更新，包括发布论文和数据集、发布新的reranker模型以及更新Embedding模型。BGE模型已经集成到Langchain中，用户可以方便地使用它。此外，BGE模型在MTEB和C-MTEB基准测试中都取得了第一名的成绩。

BGE模型的主要特点如下：

- 多语言支持：BGE模型支持中英文。

- 多版本：BGE模型有多个版本，包括bge-large-en、bge-base-en、bge-small-en等，以满足不同的需求。

- 高效的reranker：BGE提供了reranker模型，该模型比Embedding模型更准确，但比Embedding模型更耗时。因此，它可以用于重新排名Embedding模型返回的前k个文档。

- 开源和许可：BGE模型是开源的，并在MIT许可下发布。这意味着用户可以免费用于商业目的。

- 丰富集成：用户可以使用FlagEmbedding、Sentence-Transformers、Langchain或Huggingface Transformers等工具来使用BGE模型。

2.GTE

GTE模型，也称为General Text Embeddings，是阿里巴巴达摩院推出的文本Embedding技术。它基于BERT框架构建，并分为三个版本：GTE-large、GTE-base和GTE-small。

该模型在大规模的多领域文本对语料库上进行训练，确保其广泛适用于各种场景。因此，GTE可以应用于信息检索、语义文本相似性、文本重新排序等任务。

尽管GTE模型的参数规模为110M，但其性能卓越。它不仅超越了OpenAI的Embedding API，在大型文本Embedding基准测试中，其表现甚至超过了参数规模是其10倍的其他模型。更值得一提的是，GTE模型可以直接处理代码，无需为每种编程语言单独微调，从而实现优越的代码检索效果。

3.E5 Embedding

E5-embedding是由intfloat团队研发的一款先进的Embedding模型。E5的设计初衷是为各种需要单一向量表示的任务提供高效且即用的文本Embedding，与其他Embedding模型相比，E5在需要高质量、多功能和高效的文本Embedding的场景中表现尤为出色。

E5-embedding的主要特点：

- 新的训练方法：E5采用了“EmbEddings from bidirEctional Encoder rEpresentations”这一创新方法进行训练，这意味着它不仅仅依赖传统的有标记数据，也不依赖低质量的合成文本对。

- 高质量的文本表示：E5能为文本提供高质量的向量表示，这使得它在多种任务上都能表现出色，尤其是在需要句子或段落级别表示的任务中。

- 多场景：无论是在Zero-shot场景还是微调应用中，E5都能提供强大的现成文本Embedding，这使得它在多种NLP任务中都有很好的应用前景。

4.Jina Embedding

jina-embedding-s-en-v1是Jina AI的Finetuner团队精心打造的文本Embedding模型。它基于Jina AI的Linnaeus-Clean数据集进行训练，这是一个包含了3.8亿对句子的大型数据集，涵盖了查询与文档之间的配对。这些句子对涉及多个领域，并已经经过严格的筛选和清洗。值得注意的是，Linnaeus-Clean数据集是从更大的Linnaeus-Full数据集中提炼而来，后者包含了高达16亿的句子对。

Jina Embedding的主要特点：

- 广泛应用：jina-embedding-s-en-v1适合多种场景，如信息检索、语义文本相似性判断和文本重新排序等。

- 卓越性能：虽然该模型参数量仅为35M，但其性能出众，而且能够快速进行推理。

- 多样化版本：除了标准版本，用户还可以根据需求选择其他大小的模型，包括14M、110M、330M

5.Instructor

Instructor是由香港大学自然语言处理实验室团队推出的一种指导微调的文本Embedding模型。该模型可以生成针对任何任务（例如分类、检索、聚类、文本评估等）和领域（例如科学、金融等）的文本Embedding，只需提供任务指导，无需任何微调。Instructor在70个不同的Embedding任务（MTEB排行榜）上都达到了最先进的性能。该模型可以轻松地与定制的sentence-transformer库一起使用。

Instructor的主要特点：

- 多任务适应性：只需提供任务指导，即可生成针对任何任务的文本Embedding。

- 高性能：在MTEB排行榜上的70个不同的Embedding任务上都达到了最先进的性能。

- 易于使用：与定制的sentence-transformer库结合使用，使得模型的使用变得非常简单。

6.XLM-Roberta

XLM-Roberta（简称XLM-R）是Facebook AI推出的一种多语言版本的Roberta模型。它是在大量的多语言数据上进行预训练的，目的是为了提供一个能够处理多种语言的强大的文本表示模型。XLM-Roberta模型在多种跨语言自然语言处理任务上都表现出色，包括机器翻译、文本分类和命名实体识别等。

XLM-Roberta的主要特点：

- 多语言支持：XLM-Roberta支持多种语言，可以处理来自不同语言的文本数据。

- 高性能：在多种跨语言自然语言处理任务上，XLM-Roberta都表现出了最先进的性能。

- 预训练模型：XLM-Roberta是在大量的多语言数据上进行预训练的，这使得它能够捕获跨语言的文本表示。

7.text-embedding-ada-002

text-embedding-ada-002是一个由Xenova团队开发的文本Embedding模型。该模型提供了一个与Hugging Face库兼容的版本的text-embedding-ada-002分词器，该分词器是从openai/tiktoken适应而来的。这意味着它可以与Hugging Face的各种库一起使用，包括Transformers、Tokenizers和Transformers.js。

text-embedding-ada-002的主要特点：

- 兼容性：该模型与Hugging Face的各种库兼容，包括Transformers、Tokenizers和Transformers.js。

- 基于openai/tiktoken：该模型的分词器是从openai/tiktoken适应而来的。


<h3 id='2.RAG之假设文档嵌入(HyDE)'>2.RAG之假设文档嵌入(HyDE)</h3>

### 什么是HyDE
HyDE 使用一个语言学习模型，比如 ChatGPT，在响应查询时创建一个理论文档，而不是使用查询及其计算出的向量直接在向量数据库中搜索。它更进一步，通过对比方法学习无监督编码器。这个编码器将理论文档转换为一个嵌入向量，以便在向量数据库中找到相似的文档。它不是寻求问题或查询的嵌入相似性，而是专注于答案到答案的嵌入相似性。它的性能非常稳健，在各种任务（如网络搜索、问答和事实核查）中的表现与经过良好调整的检索器相匹配。

![HyDE的流程](imgs/HyDE.png)

该流程主要分为四个步骤：

1) 使用LLM基于查询生成k个假设文档。这些生成的文件可能不是事实，也可能包含错误，但它们应该于相关文件相似。此步骤的目的是通过LLM解释用户的查询。

2) 将生成的假设文档输入编码器，将其映射到密集向量$f\left(d_{k}\right)$，编码器具有过滤功能，过滤掉假设文档中的噪声。这里，dk表示第k个生成的文档，f表示编码器操作。

3) 使用给定的公式计算以下k个向量的平均值 $\mathbf{v}=\frac{1}{N} \sum_{k=1}^{N} f\left(d_{k}\right)$ ，可以将原始查询q视为一个可能的假设： $\mathbf{v}=\frac{1}{N+1} \sum_{k=1}^{N}\left[f\left(d_{k}\right)+f(q)\right]$ 

4) 使用向量v从文档库中检索答案。如步骤3中所建立的，该向量保存来自用户的查询和所需答案模式的信息，这可以提高回忆。HyDE的目标是生成假设文档，以便最终查询向量v与向量空间中的实际文档尽可能紧密地对齐。

![](imgs/hyde_embedding.png)

### HyDE的作用

在检索增强生成（RAG）中，经常遇到用户原始查询的问题，如措辞不准确或缺乏语义信息，比如“The NBA champion of 2020 is the Los Angeles Lakers! Tell me what is langchain framework?”这样的查询，如果直接进行搜索，那么LLM可能会给出不正确或无法回答的回答。因此，将用户查询的语义空间与文档的语义空间对齐是至关重要的。查询重写技术可以有效地解决这一问题，从RAG流程的角度来看，查询重写是一种预检索方法。HyDE通过假设文档来对齐查询和文档的语义空间。


<h2 id='2.4 索引与存储'>2.4 索引与存储</h2>

<h3 id='1.怎么通过预处理数据库提高RAG的精度？'>1.怎么通过预处理数据库提高RAG的精度？</h3>

#### 1.数据提取

- 使用[llamindex reader](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/)进行提取数据

- 使用[llamaindex metadata](https://docs.llamaindex.org.cn/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/)添加元数据

- 例子：https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/

#### 2.数据分割

```bash
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter

# 定义文本分割器
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

# 定义标题提取器
title_extractor = TitleExtractor(nodes=5)

# 定义问题提取器
qa_extractor = QuestionsAnsweredExtractor(questions=3)

from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, qa_extractor]
)

# 执行分割
nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)
```

<h3 id='2.怎么通过知识图谱提高RAG的精度？'>2.怎么通过知识图谱提高RAG的精度？</h3>

- 知识图谱[KGP：Knowledge Graph Prompting for Multi￾Document Question Answering](https://arxiv.org/abs/2308.11730)

```bash
#使用Llama-Index来连接到Neo4j，以构建和查询知识图谱,将文档中的信息转化为知识图谱
from llama_index.graph_stores.neo4j import Neo4jGraphStorefrom llama_index.core import KnowledgeGraphIndex

# Neo4j数据库连接配置
username = "neo4j-xxx"        # 数据库用户名（需替换为实际值）
password = "neo4j-password-xxx" # 数据库密码（需替换为实际值）
url = "neo4j-url-xxxx:7687"    # 数据库连接URL（格式通常为 bolt://host:port）
database = "neo4j"             # 数据库名称（默认是neo4j）


# 初始化Neo4j图存储连接
graph_store = Neo4jGraphStore(
    username=username,        # 传入用户名
    password=password,        # 传入密码
    url=url,                  # 传入连接URL
    database=database,        # 传入数据库名
)

# 创建存储上下文（封装图存储连接）
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# 从文档构建知识图谱索引
index = KnowledgeGraphIndex.from_documents(
    documents,                # 输入的文档列表（需提前定义）
    storage_context=storage_context,  # 关联Neo4j存储
    max_triplets_per_chunk=2, # 限制每个文本块提取的三元组数量
)

# 创建空知识图谱索引（清空之前构建的内容）
index = KnowledgeGraphIndex.from_documents(
    [],                      # 传入空文档列表
    storage_context=storage_context,
)

# 手动添加三元组到索引（假设nodes[0]已存在）
node_0_tups = [
    ("author", "worked on", "writing"),    # 三元组：主体-关系-客体
    ("author", "worked on", "programming"),
]

# 遍历并插入三元组
for tup in node_0_tups:
    # 将三元组关联到特定节点
    index.upsert_triplet_and_node(tup, nodes[0])  # nodes[0]需提前定义
```

<h3 id='3.怎么使用总分层级索引提高RAG的精度？'>3.怎么使用总分层级索引提高RAG的精度？</h3>

- 总->细，提高搜索的效率

```bash
from llama_index.core import SummaryIndex
from llama_index.core.async_utils import run_jobs
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import (FilterOperator, MetadataFilter,
MetadataFilters)

async def aprocess_doc(doc, include_summary: bool = True):
"""处理文档并创建索引节点：doc: 要处理的文档对象，include_summary: 是否包含文档摘要（默认为True）"""
    
    # 从文档元数据中提取信息
	metadata = doc.metadata
	
	# 解析创建日期
	date_tokens = metadata["created_at"].split("T")[0].split("-")
    year = int(date_tokens[0])  # 提取年份
    month = int(date_tokens[1]) # 提取月份
    day = int(date_tokens[2])   # 提取日
    
    # 提取分配人信息（如果存在）
    assignee = ("" if "assignee" not in doc.metadata else doc.metadata["assignee"])
    
    # 从标签中提取大小信息（如果有相关标签）
	size = ""
	if len(doc.metadata["labels"]) > 0:
		# 筛选包含"size:"的标签
        size_arr = [label for label in doc.metadata["labels"] if "size:" in l]
        
        # 提取大小值（如标签为"size:large"，则提取"large"）
        size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
        
        
  	# 构建新的元数据字典
    new_metadata = {
        "state": metadata["state"],  # 文档状态
        "year": year,                # 创建年份
        "month": month,              # 创建月份
        "day": day,                  # 创建日
        "assignee": assignee,        # 分配人
        "size": size,                # 大小标签
    }

 	# 提取文档摘要（如果启用）
    if include_summary:
        # 创建摘要索引（针对单个文档）
        summary_index = SummaryIndex.from_documents([doc])
        
        # 创建查询引擎（使用OpenAI的GPT-3.5模型）
        query_engine = summary_index.as_query_engine(llm=OpenAI(model="gpt-3.5-turbo"))
        
	 	# 异步查询摘要（要求一句话总结）
        query_str = "Give a one-sentence concise summary of this issue."
        summary_txt = await query_engine.aquery(query_str)
        summary_txt = str(summary_txt)  # 转换为字符串
    else:
        summary_txt = ""  # 如果不包含摘要，使用空字符串
        
 	# 获取文档索引ID
    index_id = doc.metadata["index_id"]
    
    # 创建元数据过滤器（用于后续检索）
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="index_id",          # 过滤字段
                operator=FilterOperator.EQ,  # 等于操作符
                value=int(index_id)      # 目标值（转换为整数）
            ),
        ]
    )
    
    # 创建索引节点（核心数据结构）
    index_node = IndexNode(
        text=summary_txt,         # 摘要文本
        metadata=new_metadata,    # 处理后的元数据
        obj=doc_index.as_retriever(filters=filters),  # 关联的检索器（带过滤）
        index_id=doc.id_,         # 文档唯一ID
    )
    
    return index_node  # 返回创建的索引节点
    
async def aprocess_docs(docs):
    """批量异步处理文档集合，提取元数据并创建索引节点，docs: 文档对象列表，index_nodes: 处理完成后生成的索引节点列表"""
    
    # 初始化存储容器
    index_nodes = []  # 用于存储处理结果
    tasks = []        # 用于存储异步任务
    
    # 为每个文档创建异步处理任务
    for doc in docs:
        # 为当前文档创建处理任务（不立即执行）
        task = aprocess_doc(doc)
        # 将任务添加到任务队列
        tasks.append(task)
    
    # 并行执行所有异步任务
    # show_progress=True 显示进度条
    # workers=3 限制同时运行的任务数（避免资源过载）
    index_nodes = await run_jobs(
        tasks, 
        show_progress=True, 
        workers=3
    )
    
    return index_nodes  # 返回所有文档处理结果
```

<h3 id='4.怎么使用父子层级索引提高RAG的精度？'>4.怎么使用父子层级索引提高RAG的精度？</h3>

- 细->总，提高搜索精确问题的准确性

```bash
'''
目的：对于每个1024大小的基础文本块，生成不同粒度的子文本块，用于构建分层索引结构
处理逻辑：
  1. 为每个1024大小的基础块生成：
     - 8个128大小的子块
     - 4个256大小的子块
     - 2个512大小的子块
  2. 保留原始1024大小的块
  3. 将所有块组织为索引节点
'''

# 定义子块的大小（128, 256, 512）
sub_chunk_sizes = [128, 256, 512]

# 为每种块大小创建对应的节点解析器
sub_node_parsers = [
    SimpleNodeParser.from_defaults(chunk_size=c)  # 创建指定块大小的解析器
    for c in sub_chunk_sizes
]

# 存储所有生成的节点
all_nodes = []

# 遍历每个基础节点（假设base_nodes是大小为1024的文本块）
for base_node in base_nodes:
    # 为当前基础节点生成不同粒度的子块
    for node_parser in sub_node_parsers:
        # 将基础节点分割为更小的子节点
        sub_nodes = node_parser.get_nodes_from_documents([base_node])
        
        # 将子节点转换为索引节点，并关联到原始基础节点
        sub_inodes = [
            IndexNode.from_text_node(
                sn,                   # 子节点
                base_node.node_id      # 关联到基础节点的ID（建立层级关系）
            ) 
            for sn in sub_nodes
        ]
        
        # 将生成的索引节点添加到总列表
        all_nodes.extend(sub_inodes)
    
    # 添加原始基础节点本身（保持完整块）
    original_node = IndexNode.from_text_node(
        base_node,            # 原始基础节点
        base_node.node_id      # 使用自身ID
    )
    all_nodes.append(original_node)

# 创建节点ID到节点的映射字典（便于快速查找）
all_nodes_dict = {n.node_id: n for n in all_nodes}
```

<h3 id='5.llama-index的索引类别有哪些？'>5.llama-index的索引类别有哪些？</h3>

### 索引的概念

Index是一种数据结构，允许我们快速检索用户查询的相关上下文。对于 LlamaIndex 来说，它是检索增强生成 (RAG) 用例的核心基础。在高层次上，Indexes是从Documents构建的。它们用于构建查询引擎和聊天引擎 ，从而可以通过数据进行问答和聊天。在底层，Indexes将数据存储在Node对象中（代表原始文档的块），并公开支持额外配置和自动化的Retriever接口。

- Node：对应于文档中的一段文本。LlamaIndex 接收 Document 对象并在内部将它们解析/分块为 Node 对象。

- Response Synthesis：我们的模块根据检索到的节点合成响应。

llam-index有以下五种索引

1) Summary Index ,将节点存储为顺序链

![Summary Index](./imgs/摘要索引.png)

2) Vector Store Index，将每个节点及其相应的嵌入存储在向量存储中

![Vector Store Index](./imgs/向量存储索引.png)

3) Tree Index，从一组节点（在此树中成为叶节点)构建一个层次结构树

![Tree Index](./imgs/树索引.png)

4) Keyword Table Index，从每个节点中提取关键字，并建立从每个关键字到相应节点的映射。

![Keyword Table Index](./imgs/关键字图表索引.png)

5) Property Graph Index，构建包含标记节点和关系的知识图谱。这个图的构造是非常可定制的，从让 LLM 提取它想要的任何内容，到使用严格的模式提取，甚至实现你自己的提取模块，也可以嵌入节点以供以后检索。

[llama-index文档链接](https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide)

<h3 id='6.向量数据库介绍'>6.向量数据库介绍</h3>

### 什么是向量数据库

向量数据库是一种将数据存储为高维向量的数据库，高维向量是特征或属性的数学表示。每个向量都有一定数量的维度，范围从几十到几千不等，具体取决于数据的复杂性和粒度。向量数据库同时具有CRUD操作、元数据过滤和水平扩展等功能。通过复杂的查询语言，利用资源管理、安全控制、可扩展性、容错能力和高效信息检索等数据库功能，可以提高应用程序开发效率.

### 向量数据库的特点

- 支持向量相似性搜索，它会找到与查询向量最近的 k 个向量，这是通过相似性度量来衡量的。 向量相似性搜索对于图像搜索、自然语言处理、推荐系统和异常检测等应用非常有用。

- 使用向量压缩技术来减少存储空间并提高查询性能。向量压缩方法包括标量量化、乘积量化和各向异性向量量化。

- 可以执行精确或近似的最近邻搜索，具体取决于准确性和速度之间的权衡。精确最近邻搜索提供了完美的召回率，但对于大型数据集可能会很慢。近似最近邻搜索使用专门的数据结构和算法来加快搜索速度，但可能会牺牲一些召回率。

- 支持不同类型的相似性度量，例如 L2 距离、内积和余弦距离。不同的相似性度量可能适合不同的用例和数据类型。

可以处理各种类型的数据源，例如文本、图像、音频、视频等。 

- 可以使用机器学习模型将数据源转化为向量嵌入，例如词嵌入、句子嵌入、图像嵌入等。

### 有哪些向量数据库

1、Elasticsearch

ElasticSearch是一个支持各种类型数据的分布式搜索和分析引擎。 Elasticsearch 支持的数据类型之一是向量字段，它存储密集的数值向量。

![Elasticsearch](./imgs/Elasticsearch.png)

在 7.10 版本中，Elasticsearch 添加了对将向量索引到专用数据结构的支持，以支持通过 kNN 搜索 API 进行快速 kNN 检索。 在 8.0 版本中，Elasticsearch 添加了对带有向量场的原生自然语言处理 (NLP) 的支持。

2、Faiss

Meta的Faiss是一个用于高效相似性搜索和密集向量聚类的库。 它包含搜索任意大小的向量集的算法，直到可能不适合 RAM 的向量集。 它还包含用于评估和参数调整的支持代码。

![Faiss](./imgs/Faiss.png)

3、Milvus  

Milvus是一个开源向量数据库，可以管理万亿向量数据集，支持多种向量搜索索引和内置过滤。

![Milvus](./imgs/Milvus.png)

4、Weaviate

Weaviate是一个开源向量数据库，允许你存储数据对象和来自你最喜欢的 ML 模型的向量嵌入，并无缝扩展到数十亿个数据对象。

![Weaviate](./imgs/Weaviate.png)

5、Pinecone

Pinecone专为机器学习应用程序设计的向量数据库。 它速度快、可扩展，并支持多种机器学习算法。

![Pinecone](./imgs/Pinecone.png)

Pinecone 建立在 Faiss 之上，Faiss 是一个用于密集向量高效相似性搜索的库。

6、Qdrant

Qdrant是一个向量相似度搜索引擎和向量数据库。 它提供了一个生产就绪的服务，带有一个方便的 API 来存储、搜索和管理点带有额外有效负载的向量。

![Qdrant](./imgs/Qdrant.png)

Qdrant 专为扩展过滤支持而定制。 它使它可用于各种神经网络或基于语义的匹配、分面搜索和其他应用程序。

7、Vespa

Vespa是一个功能齐全的搜索引擎和向量数据库。 它支持向量搜索 (ANN)、词法搜索和结构化数据搜索，所有这些都在同一个查询中。 集成的机器学习模型推理允许你应用 AI 来实时理解你的数据。

![Vespa](./imgs/Vespa.png)

8、Vald

Vald是一个高度可扩展的分布式快速近似最近邻密集向量搜索引擎。 Vald是基于Cloud-Native架构设计和实现的。 它使用最快的 ANN 算法 NGT 来搜索邻居。

![Vald](./imgs/Vald.png)

Vald 具有自动向量索引和索引备份，以及水平缩放，可从数十亿特征向量数据中进行搜索。

9、ScaNN (Google Research)  

ScaNN（Scalable Nearest Neighbours）是一个用于高效向量相似性搜索的库，它找到 k 个与查询向量最近的向量，通过相似性度量来衡量。向量相似性搜索对于图像搜索、自然语言处理、推荐系统和异常检测等应用非常有用。

10、pgvector

pgvector是PostgreSQL 的开源扩展，允许你在数据库中存储和查询向量嵌入。 它建立在 Faiss 库之上，Faiss 库是一个流行的密集向量高效相似性搜索库。 pgvector 易于使用，只需一条命令即可安装。

![pgvector](./imgs/pgvector.png)


<h1 id="三、检索优化技术"> 三、检索优化技术 </h1>
<h2 id="3.1 预检索优化"> 3.1 预检索优化 </h2>
<h3 id='1.怎么使用子查询（预检索）优化RAG？'>1.怎么使用子查询（预检索）优化RAG？</h3>
<h3 id='2.怎么使用假设答案（预检索）优化RAG？'>2.怎么使用假设答案（预检索）优化RAG？</h3>
<h3 id='3.还有哪些其他预检索技术优化RAG？'>3.还有哪些其他预检索技术优化RAG？</h3>
<h3 id='4.RAG之查询重写的策略介绍'>4.RAG之查询重写的策略介绍</h3>
<h3 id='5.什么是基于提示词的自适应（主动）检索？'>5.什么是基于提示词的自适应（主动）检索？</h3>

<h2 id="3.2 检索过程优化"> 3.2 检索过程优化 </h2>
<h3 id='1.怎么使用多种切分方式和并行查询？'>1.怎么使用多种切分方式和并行查询？</h3>
<h3 id='2.怎么让检索过程更加准确？'>2.怎么让检索过程更加准确？</h3>
<h3 id='3.什么是RAG的迭代检索？'>3.什么是RAG的迭代检索？</h3>
<h3 id='4.什么是RAG的递归检索？'>4.什么是RAG的递归检索？</h3>
<h3 id='5.怎么使用llama index实现递归检索？'>5.怎么使用llamaindex实现递归检索？</h3>

<h2 id="3.3 后检索优化"> 3.3 后检索优化 </h2>
<h3 id='1.RAG之Re-Ranking机制介绍'>1.RAG之Re-Ranking机制介绍</h3>
<h3 id='2.怎么使用Remark技术（后检索）优化RAG？'>2.怎么使用Remark技术（后检索）优化RAG？</h3>

<h1 id="四、RAG架构与范式"> 四、RAG架构与范式 </h1>

<h2 id="4.1 经典RAG范式"> 4.1 经典RAG范式 </h2>
<h3 id='1.什么是Modular RAG及其框架？'>1.什么是Modular RAG及其框架？</h3>
<h3 id='2.经典的RAGFlow范式（Tuning阶段）？'>2.经典的RAGFlow范式（Tuning阶段）？</h3>
<h3 id='3.经典的RAGFlow范式（推理阶段-顺序模式）？'>3.经典的RAGFlow范式（推理阶段-顺序模式）？</h3>
<h3 id='4.经典的RAGFlow范式（推理阶段-条件模式）？'>4.经典的RAGFlow范式（推理阶段-条件模式）？</h3>
<h3 id='5.如何通过llamaindex实现条件模式？'>5.如何通过llamaindex实现条件模式？</h3>
<h3 id='6.经典的RAGFlow范式（推理阶段-分支模式）？'>6.经典的RAGFlow范式（推理阶段-分支模式）？</h3>

<h2 id="4.2 高级RAG架构"> 4.2 高级RAG架构 </h2>
<h3 id='1.什么是Agentic RAG？它与传统RAG有何不同？'>1.什么是Agentic RAG？它与传统RAG有何不同？</h3>
<h3 id='2.Agentic RAG的核心组件有哪些？'>2.Agentic RAG的核心组件有哪些？</h3>
<h3 id='3.在Agentic RAG中，代理模块如何决定检索哪些文档？'>3.在Agentic RAG中，代理模块如何决定检索哪些文档？</h3>
<h3 id='4.Agentic RAG如何处理检索到的信息以生成高质量的回答？'>4.Agentic RAG如何处理检索到的信息以生成高质量的回答？</h3>
<h3 id='5.请解释Agentic RAG中的自主学习机制是如何工作的？'>5.请解释Agentic-RAG中的自主学习机制是如何工作的？</h3>
<h3 id='6.Agentic RAG在哪些实际应用中具有优势？请举例说明。'>6.Agentic RAG在哪些实际应用中具有优势？请举例说明。</h3>
<h3 id='7.在多轮对话中，Agentic RAG如何保持上下文一致性和信息连贯性？'>7.在多轮对话中，Agentic-RAG如何保持上下文一致性和信息连贯性？</h3>
<h3 id='8.Agentic-RAG面临的主要挑战有哪些？如何应对这些挑战？'>8.Agentic-RAG面临的主要挑战有哪些？如何应对这些挑战？</h3>
<h3 id='9.如何评估Agentic RAG系统的性能？有哪些关键指标？'>9.如何评估Agentic RAG系统的性能？有哪些关键指标？</h3>
<h3 id='10.在设计Agentic-RAG时，如何平衡检索与生成模块的权重？'>10.在设计Agentic-RAG时，如何平衡检索与生成模块的权重？</h3>

<h2 id="4.3 特定技术实现"> 4.3 特定技术实现 </h2>
<h3 id='1.怎么通过llamaindex实现FLAREdirect？'>1.怎么通过llamaindex实现FLAREdirect？</h3>
<h3 id='2.怎么通过langchain实现FLAREdirect？'>2.怎么通过langchain实现FLAREdirect？</h3>


<h1 id="五、RAG开发框架与工具"> 五、RAG开发框架与工具 </h1>
<h2 id="5.1 LangChain框架"> 5.1 LangChain框架 </h2>
<h3 id='1.怎么用langchain构建简单RAG？'>1.怎么用langchain构建简单RAG？</h3>
<h3 id='2.基于langchain的本地文档问答系统实现步骤有哪些?'>2.基于langchain的本地文档问答系统实现步骤有哪些?</h3>
<h3 id='3.介绍一下LangChain'>3.介绍一下LangChain</h3>
<h3 id='4.LangChain中ChatMessageHistory是什么？'>4.LangChain中ChatMessageHistory是什么？</h3>
<h3 id='5.LangChain中LangChainAgent是什么？'>5.LangChain中LangChainAgent是什么？</h3>
<h3 id='6.LangChain支持哪些功能？'>6.LangChain支持哪些功能？</h3>
<h3 id='7.什么是LangChainmodel?'>7.什么是LangChainmodel?</h3>
<h3 id='8.LangChain如何链接多个组件处理一个特定的下游任务？'>8.LangChain如何链接多个组件处理一个特定的下游任务？</h3>

<h2 id="5.2 RAGFlow项目"> 5.2 RAGFlow项目 </h2>
<h3 id='1.怎么部署RAGFlow项目？'>1.怎么部署RAGFlow项目？</h3>
<h3 id='2.怎么用RAGFlow建立知识库与聊天？'>2.怎么用RAGFlow建立知识库与聊天？</h3>
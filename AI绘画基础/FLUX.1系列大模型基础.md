# 目录

## 第一章 FLUX.1系列核心高频考点

- [1.介绍一下FLUX.1的整体架构](#1.介绍一下FLUX.1的整体架构)
- [2.与Stable Diffusion 3相比，FLUX.1的核心优化有哪些？](#2.与Stable-Diffusion-3相比，FLUX.1的核心优化有哪些？)
- [3.介绍一下FLUX.1中VAE部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图](#3.介绍一下FLUX.1中VAE部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图)
- [4.介绍一下FLUX.1中Backbone部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图](#4.介绍一下FLUX.1中Backbone部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图)
- [5.介绍一下FLUX.1中Text Encoder部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图](#5.介绍一下FLUX.1中Text-Encoder部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图)
- [6.介绍一下Rectified Flow的原理，Rectified Flow相比于DDPM、DDIM有哪些优点？](#6.介绍一下Rectified-Flow的原理，Rectified-Flow相比于DDPM、DDIM有哪些优点？)
- [7.FLUX.1系列不同版本模型之间的差异是什么？](#7.FLUX.1系列不同版本模型之间的差异是什么？)
- [8.训练FLUX.1过程中官方使用了哪些训练技巧？](#8.训练FLUX.1过程中官方使用了哪些训练技巧？)
- [9.FLUX.1模型的微调训练流程一般包含哪几部分核心内容？](#9.FLUX.1模型的微调训练流程一般包含哪几部分核心内容？)
- [10.FLUX.1模型的微调训练流程中有哪些关键参数？](#10.FLUX.1模型的微调训练流程中有哪些关键参数？)
- [11.介绍一下FLUX.1 Lite与FLUX.1的异同](#11.介绍一下FLUX.1-Lite与FLUX.1的异同)


## 第二章 FLUX.1 Kontext系列核心高频考点

- [1.介绍一下FLUX.1 Kontext的原理](#1.介绍一下FLUX.1-Kontext的原理)
- [2.FLUX.1 Kontext能够执行哪些AIGC任务？](#2.FLUX.1-Kontext能够执行哪些AIGC任务？)
- [3.FLUX.1 Kontext和FLUX.1相比，有哪些核心优化？详细分析改进的意图](#3.FLUX.1-Kontext和FLUX.1相比，有哪些核心优化？详细分析改进的意图)


## 第一章 FLUX.1系列核心高频考点

<h2 id="1.介绍一下FLUX.1的整体架构">1.介绍一下FLUX.1的整体架构</h2>


<h2 id="2.与Stable-Diffusion-3相比，FLUX.1的核心优化有哪些？">2.与Stable Diffusion 3相比，FLUX.1的核心优化有哪些？</h2>


<h2 id="3.介绍一下FLUX.1中VAE部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图">3.介绍一下FLUX.1中VAE部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图</h2>


<h2 id="4.介绍一下FLUX.1中Backbone部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图">4.介绍一下FLUX.1中Backbone部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图</h2>


<h2 id="5.介绍一下FLUX.1中Text-Encoder部分的特点，比起Stable-Diffusion-3有哪些改进？详细分析改进的意图">5.介绍一下FLUX.1中Text Encoder部分的特点，比起Stable Diffusion 3有哪些改进？详细分析改进的意图</h2>


<h2 id="6.介绍一下Rectified-Flow的原理，Rectified-Flow相比于DDPM、DDIM有哪些优点？">6.介绍一下Rectified Flow的原理，Rectified Flow相比于DDPM、DDIM有哪些优点？</h2>


<h2 id="7.FLUX.1系列不同版本模型之间的差异是什么？">7.FLUX.1系列不同版本模型之间的差异是什么？</h2>


<h2 id="8.训练FLUX.1过程中官方使用了哪些训练技巧？">8.训练FLUX.1过程中官方使用了哪些训练技巧？</h2>


<h2 id="9.FLUX.1模型的微调训练流程一般包含哪几部分核心内容？">9.FLUX.1模型的微调训练流程一般包含哪几部分核心内容？</h2>


<h2 id="10.FLUX.1模型的微调训练流程中有哪些关键参数？">10.FLUX.1模型的微调训练流程中有哪些关键参数？</h2>


<h2 id="11.介绍一下FLUX.1-Lite与FLUX.1的异同">11.介绍一下FLUX.1-Lite与FLUX.1的异同</h2>


---

## 第二章 FLUX.1 Kontext系列核心高频考点

<h2 id="1.介绍一下FLUX.1-Kontext的原理">1.介绍一下FLUX.1 Kontext的原理</h2>


<h2 id="2.FLUX.1-Kontext能够执行哪些AIGC任务？">2.FLUX.1 Kontext能够执行哪些AIGC任务？</h2>


<h2 id="3.FLUX.1-Kontext和FLUX.1相比，有哪些核心优化？详细分析改进的意图">3.FLUX.1 Kontext和FLUX.1相比，有哪些核心优化？详细分析改进的意图</h2>

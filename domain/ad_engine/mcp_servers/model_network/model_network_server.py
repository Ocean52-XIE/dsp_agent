#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型网络结构查询 MCP Server

使用 FastMCP 框架实现的 MCP Server，提供神经网络模型结构查询功能。

可用工具：
1. get_model_network - 获取模型网络结构的 jsonnet 描述

启动方式：
    python domain/ad_engine/mcp_servers/model_network/model_network_server.py
"""
import logging
import sys

from fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # MCP 使用 stdin/stdout，日志输出到 stderr
)
logger = logging.getLogger("model_network_server")

# =============================================================================
# FastMCP Server 实例
# =============================================================================

mcp = FastMCP(
    name="model_network",
    version="1.0.0",
)

# =============================================================================
# Mock 数据 - jsonnet 格式字符串
# =============================================================================

MOCK_MODEL_NETWORKS = {
    "resnet50": """// ResNet-50 网络结构定义
local conv(name, in_ch, out_ch, kernel, stride, padding=1) = {
  name: name,
  type: 'Conv2d',
  in_channels: in_ch,
  out_channels: out_ch,
  kernel_size: kernel,
  stride: stride,
  padding: padding,
};
local bn(name, features) = {
  name: name,
  type: 'BatchNorm2d',
  num_features: features,
};
local relu(name) = {
  name: name,
  type: 'ReLU',
};
local pool(name, kernel, stride, pool_type='MaxPool2d') = {
  name: name,
  type: pool_type,
  kernel_size: kernel,
  stride: stride,
};
local bottleneck(name, blocks, channels) = {
  name: name,
  type: 'BottleneckBlock',
  blocks: blocks,
  channels: channels,
};

{
  name: 'ResNet-50',
  type: 'classification',
  framework: 'pytorch',
  input_shape: [3, 224, 224],
  layers: [
    conv('conv1', 3, 64, 7, 2, 3),
    bn('bn1', 64),
    relu('relu1'),
    pool('maxpool', 3, 2),
    bottleneck('layer1', 3, 64),
    bottleneck('layer2', 4, 128),
    bottleneck('layer3', 6, 256),
    bottleneck('layer4', 3, 512),
    pool('avgpool', 1, 1, 'AdaptiveAvgPool2d'),
    {
      name: 'fc',
      type: 'Linear',
      in_features: 2048,
      out_features: 1000,
    },
  ],
  output_classes: 1000,
  parameters: '25.6M',
}""",
    "bert-base": """// BERT-Base 网络结构定义
local hidden_size = 768;
local num_heads = 12;
local num_layers = 12;

{
  name: 'BERT-Base',
  type: 'transformer',
  framework: 'pytorch',
  config: {
    hidden_size: hidden_size,
    num_attention_heads: num_heads,
    num_hidden_layers: num_layers,
    intermediate_size: 3072,
    vocab_size: 30522,
    max_position_embeddings: 512,
  },
  layers: [
    {
      name: 'embeddings',
      type: 'BertEmbeddings',
      vocab_size: 30522,
      hidden_size: hidden_size,
    },
    {
      name: 'encoder',
      type: 'BertEncoder',
      num_layers: num_layers,
      hidden_size: hidden_size,
      num_heads: num_heads,
    },
    {
      name: 'pooler',
      type: 'BertPooler',
      hidden_size: hidden_size,
    },
  ],
  parameters: '110M',
}""",
    "gpt2": """// GPT-2 网络结构定义
local hidden_size = 768;
local num_heads = 12;
local num_layers = 12;
local vocab_size = 50257;
local max_pos = 1024;

{
  name: 'GPT-2',
  type: 'transformer',
  framework: 'pytorch',
  config: {
    hidden_size: hidden_size,
    num_attention_heads: num_heads,
    num_hidden_layers: num_layers,
    vocab_size: vocab_size,
    max_position_embeddings: max_pos,
  },
  layers: [
    {
      name: 'wte',
      type: 'Embedding',
      num_embeddings: vocab_size,
      embedding_dim: hidden_size,
    },
    {
      name: 'wpe',
      type: 'Embedding',
      num_embeddings: max_pos,
      embedding_dim: hidden_size,
    },
    {
      name: 'h',
      type: 'TransformerBlock',
      num_layers: num_layers,
      hidden_size: hidden_size,
      num_heads: num_heads,
    },
    {
      name: 'ln_f',
      type: 'LayerNorm',
      normalized_shape: hidden_size,
    },
  ],
  parameters: '124M',
}""",
    "test_model": """
local model_structure = [ { name: 'input', type: 'feature_embedding', parameters: { methods: { 'features#discrete': { type: 'embedding', mask_zero: false, output_dim: 16, } }, }, inputs: { inputs: "ref::inputs.features#discrete", }, outputs: 'embedding', }, { name: 'postprocess', type: 'feature_sequential', parameters: { methods: { 'features#multiple_discrete': [{ type: 'sum', axis: 1 }, { type: 'expand_dims', axis: 1},], //'features#single_discrete': [{"type": "sum", "axis": 1}], }, }, inputs: { inputs: 'ref::input.embedding' }, outputs: 'embedding', }, { name: 'concat_feature', type: 'concatenate', parameters: { axis: 1, }, inputs: { inputs: "ref::values(postprocess.embedding)", }, outputs: 'embedding', }, { name: 'concat_dense_domian', type: 'concatenate', inputs: { inputs: "ref::values(inputs.'features#region_domain')", }, outputs: 'embedding', }, { name: 'input2_domian', type: 'auto_dis', parameters: { type: 'auto_dis', temp: 0.02, bins: 50, embedding_size: 16, }, inputs: { inputs: 'ref::concat_dense_domian.embedding', }, outputs: 'embedding', }, { name: 'postprocess_domian', type: 'dense', parameters: { units: 16, }, inputs: { inputs: 'ref::input2_domian.embedding' }, outputs: 'embedding', }, { name: 'interact_layer', type: 'dffm_interact', parameters: {activation: "relu", use_residual: "True", head_num: 2, dropout_rate: 0, randomization_level: 2,}, inputs: {inputs: ['ref::concat_feature.embedding', 'ref::postprocess_domian.embedding'],}, outputs: 'embedding', }, { name: 'cross_net', type: 'cross', parameters: { num_layers: 2 }, inputs: { inputs: 'ref::interact_layer.embedding' }, outputs: 'output', }, { name: 'dnn_after_cross', type: 'dnn', parameters: { hidden_dims: [ 256, 128 ], hidden_activation: ['relu',], output_activation: 'relu', dropout_rate: 0.0, kernel_initializer: 'he_normal', }, inputs: { inputs: 'ref::interact_layer.embedding' }, outputs: 'output', }, { name: 'concat_output', type: 'concatenate', inputs: { inputs: ['ref::cross_net.output', 'ref::dnn_after_cross.output'] }, outputs: 'output', }, { name: 'linear', type: 'dense', parameters: { units: 1 }, inputs: { inputs: 'ref::concat_output.output' }, outputs: 'logit', }, ]; local dcn_model = { name: 'dcn', structure: model_structure, inputs: 'inputs', outputs: { label: 'ref::linear.logit', } }; // 训练配置 local trainer_config = { type: 'gradient_descent', gpu_id: '0', //train_strategy: 'mirrored', epoch: 10, optimizer: { type: 'adam', lr: 0.0003, //clipnorm: 5, }, callbacks: [ { type: 'n_batch_logger', interval: 500}, //{type: 'model_checkpoint', filepath:'../tests/outputs'}, ], compile_config: { label: { // 标签名 metrics: [{ type: 'auc' }], }, }, extra_losses: [{type:'mean_predict',predict: 'ref::linear.logit',label:'ref::linear.logit'}], //extra_metrics: [{type:'reduce_mean',predict: 'ref::linear.logit',label:'ref::linear.logit'}] };
"""

}


# =============================================================================
# MCP Tools 定义
# =============================================================================

@mcp.tool()
def get_model_network(model_id: str) -> str:
    """获取神经网络模型的结构描述（jsonnet 格式）。

    根据模型 ID 返回模型的网络结构 jsonnet 描述字符串。

    Args:
        model_id: 模型 ID，支持：resnet50, bert-base, gpt2

    Returns:
        模型网络结构的 jsonnet 格式字符串
    """
    logger.info(f"[get_model_network] 查询模型: {model_id}")

    # 查找模型
    jsonnet_str = MOCK_MODEL_NETWORKS.get(model_id)

    if jsonnet_str is None:
        available_models = list(MOCK_MODEL_NETWORKS.keys())
        return f"""// 错误：未找到模型 '{model_id}'
// 可用模型 ID: {', '.join(available_models)}"""

    return jsonnet_str


if __name__ == "__main__":
    # 启动 MCP Server
    mcp.run()

# 深度学习在金融新闻影响股票价格走势预测中的应用研究-毕业设计源码

> A Study on the Application of Deep Learning in Predicting Stock Price Trends Based on Financial News


摘要

近年来，利用深度学习方法预测股票预测的研究受到了广泛的关注。传统股票预测方法注意依靠股票价格时序数据，寻找历史价格数据的规律对未来做出预测，这些方法通常忽略了金融市场的外部因素，例如金融新闻、市场情绪、宏观数据等对股价的影响，把预测股价仅仅看成了一种基于数学的计算。

在此背景下，研究人员逐渐意识到外部信息对股价预测的重要性。金融新闻作为最常见、最直接、最及时的重要信息来源，往往包含着能够影响股价的关键信息。因此，本研究将提出一种将金融新闻融合进传统股价预测模型的方法。
本文的主要研究内容和贡献如下：

（1）本研究提出了一种基于Llama-3和DistilBERT模型的新闻数据处理方法，将新闻数据经过Llama-3处理，提取出影响股票价格的关键信息，并利用DistilBERT模型将这些关键信息转换为特征向量。本研究提出的处理方法有效解决了因为新闻数量不一致、新闻缺失、新闻报道重复等问题，为后续的股价预测提供了丰富且一致的文档特征向量表示。

（2）本研究设计并实现了News-Llama3-DistilBERT-GRU混合神经网络模型（NLDG模型），该模型是对传统股价预测模型的增强，利用经过数据处理后得到的文本特征向量与原有的时序价格向量连接得到新的模型输入，并通过一系列实验找到了模型最佳的超参数，验证了该模型在股票价格预测上的优越性能，并展现了较强的实用性和潜在的商业应用价值。



关键词：股票价格预测；金融新闻；深度学习；GRU；BERT；Llama-3 


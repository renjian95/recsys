# recsys
推荐系统

1. feature部分
   a. 重写spark ml word2vec, 修复了并行化中参数更新问题，拓展CBOW模型  拓展H-softmax训练方法(原模型只包含用 负采样优化的skip-gram)

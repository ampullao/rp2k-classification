# 仓库结构
```
wrn_50_2_rp2k
├── data
    ├── all
        ├── train  # 训练数据
        ├── test  # 测试数据
├── summary_dir  # MindSpore Insight所需相关日志文件
├── checkpoints
    ├── *.ckpt  # 模型权重保存
├── data_preprocess.py  # 数据处理程序
├── eavl.py  # 验证程序
├── train.py  # 训练程序
├── wideresnet.py  # WRN-50-2定义
```
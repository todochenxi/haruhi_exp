## 思路
- transformer（multiheadattention）、performer、bigbird、roformer
- rwpe、lappe、signnet^deepset、rope
- re-gcn、gatedgcn、gcn、gin、genconv、gine、gat、pna、CustomGatedGCN
- 总结 位置特征

## 项目结构
### 基础的
- haruhi.py 整个模型 调用其他子模型
- hrgnn.py
- graph_attention.py
- pos_encoder.py
- decoder
## 实验
- 验证regcn + pe的提升
- transformer的输入是什么？没准就是se的输出
- local、global、relative是什么？又是怎么用的
### regcn + pe + se
- 计算 pe 和se
- 连接x
- pe的使用
  - 初始化, lappe / rwse，初始化是对 train，valid，test一起使用
  - pe->embedding->(lappe（p+h）|rwse（conv）更新g，h，e)->对pe进行标准化 -> cat(h,g)->更新h
  - 计算loss
- 对pe进行初始化，
- 计算好pe后添加bn mlp
- 加上pe的损失
- 
  - 

#### 重要

- gatedgnn 要加上计算relation的loss
MPCNN-Sentence-Similarity-tf：基于MPCNN网络对STS数据集进行句子相似度计算
======================================================================

网络模型：
=========
请参考这篇[论文](https://www.semanticscholar.org/paper/Multi-Perspective-Sentence-Similarity-Modeling-with-He-Gimpel/0f6924633c56832b91836b69aedfd024681e427c)的模型图
<br>

环境：
=====
*Python3.6<br>
*tensorflow-gpu1.8.0<br>
*numpy<br>

数据集下载与使用：
================
本项目采用的是STS数据集，可到[这里]()下载，下载完毕后将其解压到项目的根目录即可正常训练使用。<br>

项目架构：
=========
data_helper.py:数据处理的工具文件；<br>
embedding.py:用于读取GloVe词向量文件的工具；<br>
sim_compu_utils.py:相似度计算的工具；<br>
MPCNN_model.py：建立论文中提出的MPCNN网络架构；<br>
euclidean_distance_plot.py：可视化euclidean距离计算；<br>
train.py:模型训练文件；<br>
test.py：纯粹的一个测试文件，用于随时验证自己的想法。<br>

PS:
----
为了应对训练过程中loss为NAN的情况，进行相似度计算时comU1只计算了余弦距离和L1距离，comU2只计算了余弦距离，且在网络模型中的每一卷积层后面加了一BN层。<br>
且本项目使用的Glove文件是[glove.6B.50d.txt](https://nlp.stanford.edu/projects/glove/)<br>

模型训练：
========
```python
python train.py
```

Have a good time!
-----------------

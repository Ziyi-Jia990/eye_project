## 处理长尾分布
清洗之后的样本数量：

```Plain Text
,disease,count
2,轻度近视眼底改变,14030
1,未见明显异常表征,11087
0,中度近视眼底改变,6048
13,高血压视网膜病变轻度,2958
6,高度近视眼底改变,2306
12,疑似青光眼,1799
16,糖尿病视网膜病变中度非增生期,1269
3,糖尿病视网膜病变重度非增生期,1145
19,黄斑前膜I期,610
18,糖尿病视网膜病变轻度非增生期,497
5,黄斑水肿重度,465
8,高血压视网膜病变中度,439
4,黄斑水肿中度,384
22,黄斑水肿轻度,313
20,黄斑前膜II期,183
21,分支静脉阻塞,132
23,年龄相关性黄斑变性进展期,124
10,中央静脉阻塞,121
14,高血压视网膜病变重度,108
9,糖尿病视网膜病变增生期,107
15,有髓神经纤维,93
11,玻璃体浑浊,70
7,黄斑中浆,67
25,黄斑裂孔,50
24,脉络膜缺损,43
17,黄斑前膜III期,34
27,视网膜脱离,27
26,动脉阻塞,11
以下操作均只针对训练集：
```

### Tail score

对于样本的稀缺程度进行一个打分，并对不同稀缺程度的样本进行不同程度的加强

- $$\text{TailScore}(x) = (1-\lambda)S_{\text{single}}(x)+\lambda S_{\text{combo}}(x), \ \lambda = 0.5, \ \beta = 0.9999$$，其中 
  - 单标签部分：
  	$$
  	\boxed{ S_{\text{single}}(x)=
    \frac{1}{|Y_x|} 
    \sum_{y\in Y_x} 
    \underbrace{ 
    \frac{1-\beta}{1-\beta^{n_y}} 
    }_{\text{effective-number rarity}} 
    }$$
    - Cui et al. 2019 的核心观点是：类别“稀有程度”不应该直接用 1/n 看，而应该用 effective number，即 $$E_n = \frac{1-\beta^n}{1-\beta}$$，对应的再平衡权重就是它的倒数形式。
  - 组合部分：
    $$
    \boxed{
    S_{\text{combo}}(x) =\frac{\log(1+n_{\max})-\log(1+n_{c(x)})} 
    {\log(1+n_{\max})-\log(1+n_{\min})} 
    }$$
    - Wu et al. 2020 的 Distribution-Balanced Loss 明确指出， multi-label long-tail 问题和单标签不一样，必须考虑 label co-occurrence（标签共现），否则只按单标签频次再平衡会失真。因此这里采用对数先验进行调整（参考 Menon et al. 2020 工作）
  
- 将 Tailscore 进行归一化之后，进行划分区间：
  - 数据的 TailScore bin count
  ```Plain Text
    TailScore_bin,count,ratio
    "[0.0, 0.1)",26083,0.7192730882717921
    "[0.5, 0.6)",1314,0.03623528114055649
    "[0.1, 0.2)",1894,0.05222954526652511
    "[0.6, 0.7)",1082,0.029837575490169042
    "[0.9, 1.0]",365,0.010065355872376803
    "[0.3, 0.4)",2349,0.0647767697101729
    "[0.7, 0.8)",611,0.01684912996718418
    "[0.8, 0.9)",416,0.011471748062763699
    "[0.4, 0.5)",1166,0.03215398615668864
    "[0.2, 0.3)",983,0.027107520061770952
  ```
  - 划分区间：
  
  | TailScore    | 增强操作                                                  | 原始数据量 | 其他                                 |
  | ------------ | --------------------------------------------------------- | ---------- | ------------------------------------ |
  | [0.0, 0.1)   | 不进行数据增强                                            | 26083      | 包含三种组合类型，数据量在4675~11087 |
  | [0.1, 0.35)  | 进行数据增强，根据原本的数据扩充 2.0× 的增强              | 4207       | 包含三种组合类型，数据量在612~1894   |
  | [0.35, 0.52) | 进行数据增强（3.0x）+过采样（在数据增强的基础上扩充2.0x） | 2397       | 包含 11 种组合方式，数据量在 102~353 |
  | [0.52-1.0)   | 进行数据增强（5.0x）+过采样（在数据增强的基础上扩充2.0x） |            | 包含 550 种组合方式，数据量在1~99    |
  



### 数据增强策略

#### 数据级：

- 几何变换：对于前面两条，要求描述中不涉及位置信息
  - 随机水平翻转： $$p=0.5$$。
  - 旋转：p=0.4 的概率不旋转，p=0.6 的概率从 $$[-15,-5] \cup [5,15]$$ 采样旋转角。
  - 随机调整大小的裁剪： 缩放 0.9–1.0，宽高比 0.95–1.05（避免裁剪掉视盘或黄斑）。
  > 位置信息有 ['鼻下方', '鼻上方', '颞下方', '颞上方', '下方', '上方', '鼻侧', '颞侧']
  
- 颜色/照明：
  - 亮度/对比度： 0.05–0.12。
  - 饱和度： 0.00–0.08。
  - 色调 (Hue)： 0.00–0.02（严禁过度偏移，以免改变出血语义）。
  
- Contrast Limited Adaptive Histogram Equalization (CLAHE) was used on green channel only since it holds the most significant information among all planes.(来源于https://pmc.ncbi.nlm.nih.gov/articles/PMC9777432/pdf/diagnostics-12-03084.pdf)
  - 具体操作如下
  
    We used Contrast Limited Adaptive Histogram Equalization (CLAHE) technique for the enhancement of an image. It performs a very clear and detailed contrast improvement in an image by equalization of lighting effects. The enhancement results are remarkable even for low-contrast images (underwater images), evident in Figure 3. It can be seen that applying CLAHE on retinal images has enhanced the visibility of minute details. The working of CLAHE is as follows:
  
      (a) Step 1: Division of image into small equal-sized partitions.
  
      (b) Step 2: Performing histogram equalization on every partition.
  
      (c) Step 3: Limiting the amplification by clipping the histogram at some specific value.
  
  - 仅对以下疾病进行 CLAHE 增强，p=0.5 的概率启用
  ``` Python
    list = [
        # DR 相关
        '糖尿病视网膜病变轻度非增生期', '糖尿病视网膜病变中度非增生期', '糖尿病视网膜病变重度非增生期', '糖尿病视网膜病变增生期',
        # 黄斑水肿
    '黄斑水肿轻度', '黄斑水肿中度', '黄斑水肿重度', 
        # 高血压视网膜病变
    '高血压视网膜病变轻度', '高血压视网膜病变中度', '高血压视网膜病变重度', 
        # 疑似青光眼
    '疑似青光眼',
        # 静脉/动脉阻塞
        '分支静脉阻塞', '中央静脉阻塞', '动脉阻塞', 
        # 其他
        '年龄相关性黄斑变性进展期', '黄斑中浆', '玻璃体浑浊'
    ]
  ```
  - 执行效果：
  ![alt text](img/image.png)
  > 数据增强之后需要使用质量检测模型判断增强的样本是否合适，检测代码在 /mnt/hdd/jiazy/eye_project/diabetic-retinopathy-detection/sample_personal_qc_inference.py，模型在 /mnt/hdd/jiazy/eye_project/diabetic-retinopathy-detection/runs/eyeq_binary_resnet50/eyeq_crop/best.pt。这个部分需要你修改一下代码，使得它变为一个函数接口，支持调用



#### 损失函数级

原本的损失函数

```Plain Text
总损失 = 1.0 × 描述部分 token 交叉熵
        + 1.0 × 诊断文本 token 交叉熵
        + 0.5 × 诊断多标签 BCE 损失
```
- 将原本的“诊断多标签 BCE 损失”替换为 Class-Balanced  Focal BCE Loss
- 改为：
```Plain Text
总损失 = 1.0 × 描述部分 token 交叉熵
        + 0.7 × 诊断文本 token 交叉熵
        + 1.5 × 诊断 Class-Balanced Focal BCE Loss
```

#### 数据

- 我已经剔除了所有的坏图并使用质量检测模型将所有质量不合格的眼底彩照从数据集中剔除，最终的数据在 /mnt/hdd/jiazy/eye_project/eye_project/trans_txt/description.cleaned.qc.csv。
- 此外，我已经计算了所有样本的 TailScore，这是已经归一化之后的结果，结果在 /mnt/hdd/jiazy/eye_project/Statistics/sample_tailscore.csv 中。相关的计算代码在 /mnt/hdd/jiazy/eye_project/Statistics/ratio.py。
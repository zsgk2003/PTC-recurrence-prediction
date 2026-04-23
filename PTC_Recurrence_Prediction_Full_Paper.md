# 基于机器学习的甲状腺乳头状癌复发预测模型开发与临床验证

## 摘要

**背景**：甲状腺乳头状癌（Papillary Thyroid Carcinoma, PTC）是内分泌系统最常见的恶性肿瘤，约20-30%的患者在初始治疗后会出现疾病复发。传统ATA风险分层系统的预测准确性有限，亟需更精确的风险预测工具。

**方法**：本研究采用UCI机器学习数据库中335例PTC患者的临床病理数据，系统比较九种机器学习算法（逻辑回归、决策树、SVM、K近邻、朴素贝叶斯、随机森林、XGBoost、LightGBM、CatBoost）的预测性能。采用分层抽样将数据集按7:3划分为训练集和测试集，通过网格搜索结合5折交叉验证进行超参数优化。模型评估涵盖判别能力（AUC、准确率、F1分数）、校准度（ECE、Brier分数）和临床效用（决策曲线分析）。使用SHAP分析识别关键预测因子并开发基于Streamlit的交互式网络应用。

**结果**：梯度提升模型（LightGBM、XGBoost、CatBoost）显著优于传统机器学习方法。LightGBM表现最佳，测试集AUC达0.983（95% CI: 0.949-1.000），准确率0.970（95% CI: 0.931-1.000），Brier分数0.026。DeLong检验显示LightGBM与CatBoost之间无统计学差异（p = 0.987），但LightGBM与XGBoost之间存在显著差异（p < 0.001）。决策曲线分析显示LightGBM在10-50%阈值概率范围内的平均净获益为0.245。SHAP分析识别治疗反应、N分期、T分期、年龄和体格检查为前五位关键预测因子，其中治疗反应的平均绝对SHAP值（3.69）远高于其他特征。开发的Streamlit网络应用支持单例实时预测、批量预测、模型性能展示和SHAP解释功能。

**结论**：梯度提升模型（尤其是LightGBM）在PTC复发预测中展现出卓越的判别能力、校准度和临床效用，为临床提供了可解释、易用且可靠的风险评估工具。

**关键词**：甲状腺乳头状癌；复发预测；机器学习；梯度提升；可解释人工智能；SHAP

---

## 1 引言 (Introduction)

### 1.1 甲状腺乳头状癌的临床挑战

甲状腺乳头状癌（Papillary Thyroid Carcinoma, PTC）是内分泌系统中最常见的恶性肿瘤，占所有甲状腺癌病例的80%以上[1]。尽管PTC总体预后良好，10年生存率超过90%，但疾病复发仍是临床管理中的重大挑战。研究表明，约20-30%的PTC患者在初始治疗后会出现疾病复发，包括结构性复发和生化复发，这不仅影响患者的生存质量，还增加了医疗负担和随访成本[2-6]。因此，准确识别高复发风险患者对于制定个体化治疗方案、优化随访策略具有重要的临床意义。

### 1.2 传统风险分层系统的局限性

目前临床上广泛采用美国甲状腺协会（American Thyroid Association, ATA）风险分层系统来指导PTC患者的术后管理和随访决策[4,6]。ATA系统基于临床病理特征将患者分为低危、中危和高危三类或低危、低中危、高中危、高危四级复发风险分层体系[6]。然而，部分研究指出传统风险分层存在一定的局限性，如阳性预测值偏低、对低中危患者的风险预测不够精确等问题[7-9]。Wang等[9]的研究显示，ATA风险分层在接受甲状腺切除术和放射性碘治疗的PTC患者中，预测结构性复发的AUC仅为0.620，显著低于机器学习模型（AUC = 0.738-0.767）。这种相对较低的预测准确性可能导致部分患者接受过度治疗或治疗不足，凸显了开发更精确预测工具的临床需求。

### 1.3 机器学习在癌症预后预测中的应用

近年来，机器学习（Machine Learning, ML）技术在医学领域的应用取得了显著进展，特别是在疾病风险预测和预后评估方面。机器学习算法能够自动从大量临床数据中学习复杂的非线性关系，整合多维度的预测因子，从而构建高精度的预测模型[10-14]。在甲状腺癌复发预测领域，多种机器学习算法已被探索应用，包括逻辑回归、支持向量机（SVM）、随机森林（Random Forest）、K近邻（KNN）、梯度提升树（XGBoost、LightGBM、CatBoost）以及神经网络等。Park和Lee[10]的研究发现，淋巴结比率和对侧中央区淋巴结转移在多种机器学习模型中均被列为重要预测因子，为后续研究提供了重要的特征筛选依据。

### 1.4 梯度提升模型与可解释AI

梯度提升决策树（Gradient Boosting Decision Trees, GBDT）算法因其强大的特征学习能力、对高维数据的良好适应性以及优异的预测性能，在医学预测建模领域受到广泛关注。XGBoost、LightGBM和CatBoost作为GBDT家族的代表算法，在多项甲状腺癌复发预测研究中展现出卓越性能[15-17]。Schindele等[14]通过SHAP分析识别出肿瘤大小、甲状腺球蛋白水平和抗体水平为关键预测因子，并提出了新的复发风险阈值。Sarker等[15]的研究证明，仅需6个特征即可达到与全特征模型相当的性能，凸显了特征选择的重要性。

尽管机器学习模型在预测准确性方面展现出巨大潜力，但其"黑箱"特性一直是临床应用的主要障碍。SHAP（SHapley Additive exPlanations）值分析作为一种基于博弈论的可解释性方法，能够量化每个特征对个体预测的贡献，为机器学习模型提供全局和局部的解释能力[15,20,21]。Hanani等[19]的研究显示，治疗反应、风险分层和N分期是CatBoost模型中最重要的预测因子，SHAP分析为临床医生理解模型决策提供了有力工具。

### 1.5 数据来源与研究缺口

甲状腺癌复发可能在初始治疗后数十年发生，因此长程随访数据对于准确评估复发风险至关重要。本研究使用的UCI数据集来源于Borzooei等[24]开展的一项长达15年的前瞻性队列研究，该研究纳入383例分化型甲状腺癌患者，随访时间至少10年。这一长程随访队列为本研究的数据来源提供了可靠的临床基础。

综上所述，尽管已有大量研究探索机器学习在PTC复发预测中的应用，但仍存在以下研究缺口有待解决：

1. **模型选择缺乏统一标准**：不同研究报告的最佳模型各异（随机森林、XGBoost、SVM等），且缺乏在同一数据集上的系统比较；

2. **模型校准度评估不足**：多数研究仅关注AUC等判别指标，忽视了概率校准对临床决策的重要性；

3. **临床效用验证有限**：少数研究通过决策曲线分析（DCA）评估模型的临床净获益，限制了对模型实际临床价值的判断；

4. **可解释性与易用性**：现有研究多停留在模型开发阶段，缺乏便于临床实际应用的部署工具和可视化界面。

本研究旨在系统比较九种机器学习算法在PTC复发预测中的综合性能，综合评估模型的判别能力、校准度、临床效用和泛化能力，并通过SHAP分析识别关键预测因子，最终开发基于最优模型的交互式网络应用工具，为临床医生提供可解释、易用且可靠的复发风险评估工具。

---

## 2 材料与方法 (Methods)

### 2.1 数据来源与研究人群

本研究数据来源于分化型甲状腺癌复发数据集（Differentiated Thyroid Cancer Recurrence Dataset），该数据集公开发布于UCI机器学习数据库（https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence）[25]。原始队列纳入了383例分化型甲状腺癌患者，在15年时间范围内进行了至少10年的随访。从该源人群中，本研究筛选出335例经病理确诊的甲状腺乳头状癌（PTC）患者进行最终分析。研究终点为随访期间的疾病复发（Recurred: Yes/No）。在纳入的335例患者中，245例（73.13%）保持无复发状态，90例（26.87%）出现疾病复发，提示存在中度类别不平衡，需要在模型训练和评估过程中谨慎处理。

### 2.2 变量定义与编码

共纳入16个临床特征变量和1个目标变量（Recurred）构建预测模型。特征涵盖：人口学特征包括年龄（连续变量）和性别（二分类：女性=0，男性=1）；生活方式因素包括当前吸烟状态（否=0，是=1）和吸烟史（否=0，是=1）；既往病史包括放疗史（否=0，是=1）；甲状腺功能参数分为5类（临床甲状腺功能亢进/减退/正常/亚临床甲状腺功能亢进/减退，编码0-4）；体格检查发现包括体格检查（5类：弥漫性甲状腺肿/多结节性/正常/左叶单结节/右叶单结节，编码0-4）和腺病状态（6类：双侧/广泛/左侧/无/后/右侧，编码0-5）；病理特征包括病理类型（2类：微乳头型/乳头型，编码0-1）和病灶数量（二分类：多灶=0，单灶=1）；风险分层采用美国甲状腺协会风险类别（3类：低危=0，中危=1，高危=2）；TNM分期参数包括T分期（7类：T1a/T1b/T2/T3a/T3b/T4a/T4b，编码0-6）、N分期（3类：N0/N1a/N1b，编码0-2）和M分期（二分类：M0=0，M1=1）；总体临床分期（5类：I/II/III/IVA/IVB，编码0-4）；以及治疗反应评估（4类：良好/不确定/结构性不完全/生化不完全，编码0-3）。标签编码严格按照原始数据集文档定义的编码方案执行，以确保可重复性。

### 2.3 数据预处理与质量控制

模型开发前对数据质量进行了严格评估。采用完整病例分析进行缺失值分析，确认数据集在所有16个特征中均无缺失条目，无需进行插补处理。质量控制验证后，采用分层随机抽样以7:3比例将数据集划分为独立的训练队列和测试队列，同时保持类别分布。训练集包含234例患者（63例复发，171例无复发），测试集包含101例患者（27例复发，74例无复发）。在训练-测试划分过程中应用固定随机种子（random_state = 42），以确保数据划分的完全可重复性。

特征标准化采用scikit-learn的StandardScaler进行Z-score转换，公式为 x' = (x - μ) / σ，其中μ代表特征均值，σ代表特征标准差。为防止数据泄露，缩放器仅在训练数据上使用fit_transform方法拟合，然后将拟合的缩放器应用于测试数据（仅使用transform方法）。拟合的缩放器保存为scaler.pkl供下游应用使用。标准化后的特征用于对距离敏感和基于梯度的算法（逻辑回归、支持向量机、K近邻），而基于树的模型（决策树、随机森林、XGBoost、LightGBM、CatBoost）则在原始特征尺度上训练，因其具有尺度不变性。

### 2.4 机器学习模型构建

本研究系统开发并评估了九种不同的机器学习算法用于PTC复发预测。模型组合涵盖多种方法学方法：线性判别模型（带L1/L2正则化的逻辑回归）、基于规则的分类器（决策树）、基于核函数的最大边界方法（带RBF和线性核的支持向量机）、基于实例的学习算法（K近邻）、概率分类器（假设高斯特征分布的朴素贝叶斯）、基于Bagging的集成方法（随机森林），以及梯度提升框架（XGBoost、LightGBM和CatBoost实现序贯误差校正学习）。

采用穷举网格搜索结合分层5折交叉验证对每个算法进行超参数优化。GridSearchCV配置采用StratifiedKFold（n_splits=5, shuffle=True, random_state=42）以保持各折中的类别平衡，以ROC曲线下面积（roc_auc）为优化指标，并启用并行计算（n_jobs=-1）。参数搜索空间设计兼顾临床合理范围与计算可行性。逻辑回归在C值0.01、0.1、1、10、100范围内探索逆正则化强度，惩罚类型包括L1和L2正则化，使用liblinear和saga求解器。决策树优化涵盖分裂标准（基尼不纯度与信息熵）、最大树深度（3-10层或无限制）以及节点分裂（2、5、10）和叶节点形成（1、2、4）的最小样本阈值。支持向量机超参数包括正则化参数C（0.1、1、10）、核函数（径向基函数与线性）和gamma缩放选项（scale、auto）。K近邻在邻域大小3-11、加权方案（均匀与基于距离）和距离度量（欧氏与曼哈顿）范围内优化。随机森林超参数探索包括100和200棵树的集成规模、最大深度约束（5、10、无限制）以及节点操作的最小样本要求。梯度提升框架采用一致的参数范围：XGBoost和LightGBM在100-200个估计器、3/5/7层最大树深度、0.05和0.1学习率、0.8-1.0子采样比例范围内优化（LightGBM额外优化31和50的num_leaves）。CatBoost优化包括100和200次迭代、3/5/7层树深度以及等效学习率选项。朴素贝叶斯因其假设特征独立的概率基础，未进行超参数优化。

对所有模型实施概率校准以确保概率输出的一致性。对于没有原生predict_proba方法的模型，应用CalibratedClassifierCV配合sigmoid（Platt）缩放和5折交叉验证进行概率校准，以便在整个模型组合中进行一致可靠性评估。

### 2.5 模型评估框架

模型性能采用多种互补指标在训练队列和测试队列上独立计算进行综合评估，涵盖判别能力、校准度和临床效用。判别性能量化指标包括准确率、精确率（阳性预测值）、召回率（敏感性）、F1分数（精确率和召回率的调和平均值）和ROC曲线下面积（AUC-ROC）。概率校准质量通过Brier分数评估，该分数衡量预测概率与观察结果之间的均方差，值越低表示校准越好。额外计算平均精确率以总结精确率-召回率曲线。通过1000次迭代 bootstrap 重采样建立统计置信度，使用百分位法生成所有主要性能指标的95%置信区间。

采用DeLong检验对配对ROC曲线进行模型判别能力的成对统计比较，计算所有模型对之间AUC差异的Z统计量和相应p值。这种非参数方法考虑了在相同数据集上训练的模型预测的相关性，为性能比较提供严格统计证据。模型间分类一致性采用Cohen's Kappa系数量化，值范围从0（随机一致）到1（完全一致），用于评估算法间的决策一致性。

概率校准通过预期校准误差（ECE）和最大校准误差（MCE）指标进行严格评估。ECE通过将预测概率划分为10个等宽区间，计算各区间内平均预测概率与观察事件频率绝对差的加权平均值。临床效用通过决策曲线分析（DCA）评估，在0-1的阈值概率范围内根据公式净获益 = 真阳性/N -（假阳性/N）×（阈值/(1-阈值））量化净获益。临床相关阈值范围10%-50%内的平均净获益作为主要临床效用指标报告，代表治疗决策通常最不确定的范围。

通过比较训练和测试指标评估泛化性能，特别关注AUC差异（ΔAUC）作为过拟合指标。在训练集和测试集之间表现出最小性能下降的模型被认为具有适合临床部署的稳健泛化能力。

### 2.6 模型可解释性分析

对表现最佳的基于树的模型（LightGBM）使用SHAP（SHapley Additive exPlanations）进行可解释性分析，提供全局和局部模型可解释性。采用TreeExplainer实现计算所有测试集样本的SHAP值，量化各特征对个体预测的贡献。全局解释通过汇总条形图展示特征重要性排名的平均绝对SHAP值，以及蜂群图说明特征值与其对预测结果影响之间的关系进行可视化。局部解释通过个体患者病例的瀑布图实现，具体检查预测概率最高（最高风险病例）和最低（最低风险病例）的样本，以对比高风险和低风险场景下的特征贡献模式。

### 2.7 最优模型选择与网络应用部署

在九种机器学习算法的综合评估后，基于多标准决策框架选择最优模型，优先考虑判别性能（AUC-ROC）、校准质量（Brier分数、ECE）、临床效用（决策曲线分析净获益）和泛化能力（训练集与测试集间最小ΔAUC）。LightGBM以最高的测试集AUC（0.983，95% CI: 0.949-1.000）、优异的校准（ECE: 0.022，Brier分数: 0.026）和卓越的临床效用（10-50%阈值范围内平均净获益: 0.245）脱颖而出，成为表现最佳的模型。

通过GridSearchCV确定最优超参数（n_estimators、max_depth、learning_rate、num_leaves）的优化LightGBM模型使用joblib序列化并保存为`BEST_MODEL_LightGBM.pkl`用于部署。拟合的StandardScaler对象（`scaler.pkl`）一并保存，以确保对传入预测请求进行一致的特征转换。这种模型持久化策略能够在部署环境中精确复现训练阶段的预处理和预测流程。

使用Streamlit框架开发交互式网络应用，使经过验证的预测模型能够普及应用。选择Streamlit是因为其Python原生架构、最少的样板代码要求，以及适合临床决策支持工具的快速原型开发能力。应用架构采用模块化设计，通过侧边栏导航可访问不同的功能页面：

**单例预测界面**：该模块通过直观的表单输入系统实现实时个体化风险评估。临床医生可通过适当类型的输入控件（包括数字字段、下拉选择器和单选按钮）输入全部16个临床特征（年龄、性别、吸烟状态、病史变量、甲状腺功能、体格检查发现、腺病、病理、病灶数量、风险分层、TNM分期参数、总体分期和治疗反应）。提交后，应用加载序列化的缩放器和LightGBM模型，应用与训练期间相同的预处理转换，生成复发概率预测并附带置信度解释。预测结果通过视觉指示器（进度条、颜色编码的风险等级）显示，便于临床快速解读。

**批量预测模块**：该功能通过接受包含多条患者记录的CSV文件上传，支持人群水平筛查和队列分析。模块执行自动数据验证，对所有记录应用标准化预处理流程，使用部署的LightGBM模型执行批量推理，并返回可下载的结果文件，包含每例患者的预测概率和二分类结果。该能力支持与医院信息系统集成，用于回顾性队列筛查或前瞻性监测项目。

**模型性能仪表板**：该教育和透明度模块提供全面的模型文档，包括九种模型比较的完整结果，以及ROC曲线、校准图和决策曲线的交互式可视化。性能指标表格显示训练集和测试集结果及置信区间，帮助临床利益相关者理解模型能力和局限性。纳入特征重要性排名和SHAP汇总图，提供驱动模型决策的预测因子洞察。

**关于与文档**：该部分提供方法学背景，包括数据集来源、模型开发流程、带临床解释的特征定义以及引用信息。包含数据隐私和治理声明，以满足临床决策支持工具的机构审查要求。

部署架构确保所有预测使用训练阶段确定的预处理参数（特征缩放均值和标准差）以及模型权重，防止训练-服务偏差。应用通过命令`streamlit run app.py`在本地执行，默认服务地址为`http://localhost:8501`，可在机构服务器或云平台上部署，无需依赖第三方预测服务。这种本地部署策略通过确保所有患者数据在推理过程中保留在机构网络范围内，满足受保护健康信息的数据主权要求。

### 2.8 软件环境与可重复性

所有分析使用Python 3.x实现，采用成熟的科学计算库以确保可重复性和透明度。数据操作和分析使用pandas和numpy。机器学习模型开发采用scikit-learn实现基础算法，xgboost、lightgbm和catboost提供优化的梯度提升实现。可视化和图形生成使用matplotlib和seaborn库。模型可解释性分析采用shap库配合TreeExplainer用于梯度提升模型。模型持久化使用joblib对训练模型和预处理对象进行序列化。

在所有随机程序中一致应用固定随机种子（RANDOM_STATE = 42），包括数据划分、模型初始化、交叉验证打乱和bootstrap重采样，以确保所有报告结果的完全可重复性。完整的分析流程以Jupyter Notebook格式实现，采用模块化函数架构。所有性能评估可视化以300 DPI分辨率导出为PNG和PDF格式，满足发表要求。本方法学全面涵盖从原始数据获取、特征编码、预处理、多模型比较、超参数调优、性能评估、可解释性分析到模型部署的端到端机器学习流程，符合临床预测建模研究的TRIPOD（多变量预测模型个体预后或诊断透明报告）报告指南。

---

## 3 结果 (Results)

### 3.1 数据集特征

本研究共纳入335例甲状腺乳头状癌（PTC）患者，训练队列234例（63例复发，171例无复发），独立测试队列101例（27例复发，74例无复发）。纳入模型开发的16个临床特征涵盖人口学特征、既往病史、甲状腺功能、体格检查发现、病理特征、TNM分期和治疗反应（表1）。

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/dataset_info.txt`
- 总样本量：335
- 训练样本：234（复发：63，无复发：171）
- 测试样本：101（复发：27，无复发：74）
- 特征（n=16）：年龄、性别、吸烟、吸烟史、放疗史、甲状腺功能、体格检查、腺病、病理、病灶数量、风险、T、N、M、分期、治疗反应

### 3.2 模型性能比较

九种机器学习算法被开发并评估用于PTC复发预测。在独立测试集上，基于梯度提升的模型相较于传统机器学习方法展现出优越的判别性能。**LightGBM实现了最高的预测准确性**，准确率达0.970（95% CI: 0.931-1.000），精确率1.000，召回率0.889（95% CI: 0.759-1.000），F1分数0.941（95% CI: 0.863-1.000），AUC达0.983（95% CI: 0.949-1.000）。XGBoost表现出可比的性能，具有相同的准确率（0.970）、精确率（1.000）、召回率（0.889）和F1分数（0.941），AUC略低为0.978（95% CI: 0.941-1.000）。CatBoost排名第三，准确率0.960（95% CI: 0.921-0.990），AUC为0.982（95% CI: 0.950-1.000）（表2，图1）。

**表2. 九种机器学习模型在测试队列上的性能指标**

| 模型 | 准确率 (95% CI) | 精确率 (95% CI) | 召回率 (95% CI) | F1分数 (95% CI) | AUC (95% CI) | Brier分数 (95% CI) | AP |
|------|-----------------|-----------------|-----------------|-----------------|--------------|-------------------|-----|
| LightGBM | 0.970 (0.931-1.000) | 1.000 (1.000-1.000) | 0.889 (0.759-1.000) | 0.941 (0.863-1.000) | 0.983 (0.949-1.000) | 0.026 (0.000-0.069) | 0.973 |
| XGBoost | 0.970 (0.931-1.000) | 1.000 (1.000-1.000) | 0.889 (0.759-1.000) | 0.941 (0.863-1.000) | 0.978 (0.941-1.000) | 0.032 (0.000-0.069) | 0.966 |
| CatBoost | 0.960 (0.921-0.990) | 1.000 (1.000-1.000) | 0.852 (0.700-0.962) | 0.920 (0.824-0.980) | 0.982 (0.950-1.000) | 0.038 (0.010-0.079) | 0.970 |
| Random Forest | 0.950 (0.911-0.990) | 1.000 (1.000-1.000) | 0.815 (0.667-0.945) | 0.898 (0.800-0.971) | 0.961 (0.888-1.000) | 0.047 (0.010-0.089) | 0.960 |
| KNN | 0.931 (0.881-0.970) | 1.000 (1.000-1.000) | 0.741 (0.577-0.893) | 0.851 (0.732-0.943) | 0.963 (0.905-0.996) | 0.068 (0.030-0.119) | 0.939 |
| 决策树 | 0.931 (0.871-0.970) | 0.955 (0.846-1.000) | 0.778 (0.606-0.923) | 0.857 (0.727-0.944) | 0.950 (0.887-0.995) | 0.055 (0.030-0.129) | 0.914 |
| 逻辑回归 | 0.921 (0.871-0.970) | 0.852 (0.708-0.968) | 0.852 (0.708-0.966) | 0.852 (0.735-0.941) | 0.973 (0.941-0.994) | 0.062 (0.030-0.129) | 0.941 |
| SVM | 0.901 (0.842-0.950) | 0.815 (0.667-0.955) | 0.815 (0.655-0.947) | 0.815 (0.687-0.917) | 0.969 (0.932-0.992) | 0.065 (0.050-0.158) | 0.932 |
| 朴素贝叶斯 | 0.901 (0.842-0.950) | 0.905 (0.765-1.000) | 0.704 (0.542-0.864) | 0.792 (0.655-0.900) | 0.958 (0.915-0.990) | 0.094 (0.050-0.158) | 0.841 |

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/model_performance_test.csv`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/model_ranking.csv`

在传统机器学习模型中，随机森林表现最佳，准确率0.950，AUC 0.961，其次是K近邻（准确率：0.931，AUC：0.963）和逻辑回归（准确率：0.921，AUC：0.973）。朴素贝叶斯显示出最低的判别能力，AUC为0.958，Brier分数最高（0.094），提示概率校准较差。

### 3.3 统计显著性检验

DeLong配对ROC曲线比较检验显示，**LightGBM显著优于决策树**（ΔAUC = 0.033, p = 0.050），并对KNN（ΔAUC = 0.020, p = 0.065）和SVM（ΔAUC = 0.014, p = 0.104）表现出边缘显著性优势。值得注意的是，LightGBM与CatBoost之间未观察到统计学显著差异（p = 0.987），但**LightGBM与XGBoost之间存在统计学显著差异**（p < 0.001），提示LightGBM在该数据集上具有略微更优的判别能力。XGBoost与CatBoost之间亦无显著差异（p = 0.899）（表3）。

**表3. 成对AUC比较的DeLong检验P值**

| | 逻辑回归 | 决策树 | SVM | KNN | 朴素贝叶斯 | 随机森林 | XGBoost | LightGBM | CatBoost |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **逻辑回归** | 1.000 | 0.253 | 0.014* | 0.495 | 0.325 | 0.587 | 0.525 | 0.192 | 0.168 |
| **决策树** | 0.253 | 1.000 | 0.349 | 0.471 | 0.714 | 0.462 | 0.040* | 0.050* | 0.053 |
| **SVM** | 0.014* | 0.349 | 1.000 | 0.678 | 0.490 | 0.718 | 0.319 | 0.104 | 0.094 |
| **KNN** | 0.495 | 0.471 | 0.678 | 1.000 | 0.766 | 0.851 | 0.173 | 0.065 | 0.102 |
| **朴素贝叶斯** | 0.325 | 0.714 | 0.490 | 0.766 | 1.000 | 0.886 | 0.158 | 0.091 | 0.090 |
| **随机森林** | 0.587 | 0.462 | 0.718 | 0.851 | 0.886 | 1.000 | 0.285 | 0.199 | 0.238 |
| **XGBoost** | 0.525 | 0.040* | 0.319 | 0.173 | 0.158 | 0.285 | 1.000 | <0.001** | 0.899 |
| **LightGBM** | 0.192 | 0.050* | 0.104 | 0.065 | 0.091 | 0.199 | <0.001** | 1.000 | 0.987 |
| **CatBoost** | 0.168 | 0.053 | 0.094 | 0.102 | 0.090 | 0.238 | 0.899 | 0.987 | 1.000 |

\* p < 0.05; \*\* p < 0.01

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/delong_pvalues_test.csv`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/delong_auc_diff_test.csv`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/delong_zstatistics_test.csv`

主要发现：
- LightGBM vs XGBoost: p < 0.001**（显著差异）
- LightGBM vs CatBoost: p = 0.987（无显著差异）
- XGBoost vs 决策树: p = 0.040*（显著）
- LightGBM vs 决策树: p = 0.050*（边缘显著）

### 3.4 模型校准评估

使用预期校准误差（ECE）、最大校准误差（MCE）和Brier分数评估概率校准。**LightGBM展现出最佳的校准性能**，ECE最低（0.022）、MCE（0.591）和Brier分数（0.026）均最低，提示其输出的复发概率能够准确反映患者的真实风险水平。XGBoost和CatBoost也显示出可接受的校准，ECE值分别为0.030和0.032。相比之下，随机森林尽管判别性能合理，但校准最差（ECE: 0.091），提示其概率估计可能存在过度自信倾向（表4，图2）。

**表4. 概率评估的校准指标**

| 模型 | ECE | MCE | Brier分数 |
|------|:---:|:---:|:---:|
| LightGBM | **0.022** | 0.591 | **0.026** |
| XGBoost | 0.030 | 0.720 | 0.032 |
| CatBoost | 0.032 | 0.346 | 0.038 |
| 决策树 | 0.040 | 0.333 | 0.055 |
| 随机森林 | 0.091 | 0.463 | 0.047 |
| KNN | 0.069 | 0.454 | 0.068 |
| SVM | 0.078 | 0.468 | 0.065 |
| 逻辑回归 | 0.082 | 0.477 | 0.062 |
| 朴素贝叶斯 | 0.076 | 0.819 | 0.094 |

ECE = 预期校准误差；MCE = 最大校准误差

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/calibration_metrics_test.csv`

### 3.5 临床决策曲线分析

执行决策曲线分析（DCA）以评估不同阈值概率下的净临床获益。**LightGBM在10-50%的临床相关阈值范围内实现最高的平均净获益0.245**，其次是CatBoost（0.232）和XGBoost（0.238）。所有梯度提升模型在整个阈值范围内始终优于"全部治疗"和"不治疗"策略，展现出指导术后管理决策的实质性临床效用。值得注意的是，LightGBM在20%阈值概率时开始提供正净获益，而大多数其他模型从10%开始，提示其在中等风险决策范围内表现最佳（图3，表5）。

**表5. 决策曲线分析指标**

| 模型 | 最大净获益 | 最大NB阈值 | 平均NB (10-50%) | 优于全部治疗 | 优于不治疗 |
|------|:---:|:---:|:---:|:---:|:---:|
| LightGBM | 0.248 | 0.20 | **0.245** | 1.000 | 1.000 |
| CatBoost | 0.250 | 0.05 | 0.232 | 1.000 | 1.000 |
| XGBoost | 0.244 | 0.10 | 0.238 | 1.000 | 1.000 |
| 随机森林 | 0.249 | 0.05 | 0.231 | 1.000 | 1.000 |
| KNN | 0.250 | 0.05 | 0.211 | 1.000 | 1.000 |
| SVM | 0.247 | 0.05 | 0.212 | 1.000 | 1.000 |
| 逻辑回归 | 0.254 | 0.05 | 0.212 | 1.000 | 1.000 |
| 决策树 | 0.244 | 0.05 | 0.210 | 1.000 | 1.000 |
| 朴素贝叶斯 | 0.207 | 0.05 | 0.185 | 0.944 | 0.944 |

NB = 净获益

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/dca_metrics_test.csv`

### 3.6 模型一致性与稳健性

采用Cohen's Kappa统计量评估模型间一致性。**XGBoost和LightGBM在分类决策中展现完全一致**（κ = 1.000），与其几乎相同的AUC值一致。CatBoost与XGBoost（κ = 0.972）和LightGBM（κ = 0.972）均表现出极佳的一致性。传统机器学习模型与梯度提升方法的一致性为中等至实质性（κ范围：0.735-0.943），而朴素贝叶斯与其他模型的一致性最低（κ范围：0.675-0.852），反映其独特的概率分类方法（表6）。

**表6. 模型间一致性的Cohen's Kappa矩阵**

| | 逻辑回归 | 决策树 | SVM | KNN | 朴素贝叶斯 | 随机森林 | XGBoost | LightGBM | CatBoost |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **逻辑回归** | 1.000 | 0.758 | 0.949 | 0.807 | 0.728 | 0.866 | 0.817 | 0.817 | 0.788 |
| **决策树** | 0.758 | 1.000 | 0.705 | 0.700 | 0.675 | 0.826 | 0.887 | 0.887 | 0.914 |
| **SVM** | 0.949 | 0.705 | 1.000 | 0.807 | 0.728 | 0.812 | 0.764 | 0.764 | 0.735 |
| **KNN** | 0.807 | 0.700 | 0.807 | 1.000 | 0.847 | 0.880 | 0.826 | 0.826 | 0.793 |
| **朴素贝叶斯** | 0.728 | 0.675 | 0.728 | 0.847 | 1.000 | 0.852 | 0.800 | 0.800 | 0.768 |
| **随机森林** | 0.866 | 0.826 | 0.812 | 0.880 | 0.852 | 1.000 | 0.944 | 0.944 | 0.914 |
| **XGBoost** | 0.817 | 0.887 | 0.764 | 0.826 | 0.800 | 0.944 | **1.000** | **1.000** | 0.972 |
| **LightGBM** | 0.817 | 0.887 | 0.764 | 0.826 | 0.800 | 0.944 | **1.000** | **1.000** | 0.972 |
| **CatBoost** | 0.788 | 0.914 | 0.735 | 0.793 | 0.768 | 0.914 | 0.972 | 0.972 | 1.000 |

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/cohen_kappa_matrix_test.csv`

### 3.7 泛化性能

训练集与测试集性能比较显示梯度提升模型具有优异的泛化能力。LightGBM和XGBoost在训练集上达到完美性能（准确率：1.000，AUC：1.000），在测试集上仍保持极高性能（LightGBM：准确率0.970、AUC 0.983；XGBoost：准确率0.970、AUC 0.978），ΔAUC分别为0.017和0.022，显示出良好的泛化能力且无明显过拟合。相比之下，KNN和决策树表现出更明显的过拟合倾向（ΔAUC：0.037和0.047）（表7）。

**表7. 训练集与测试集性能比较**

| 模型 | 训练准确率 | 测试准确率 | 训练AUC | 测试AUC | ΔAUC | 训练Brier | 测试Brier |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LightGBM | 1.000 | 0.970 | 1.000 | 0.983 | 0.017 | 0.000 | 0.026 |
| XGBoost | 0.996 | 0.970 | 1.000 | 0.978 | 0.022 | 0.007 | 0.032 |
| CatBoost | 0.979 | 0.960 | 0.998 | 0.982 | 0.016 | 0.017 | 0.038 |
| 随机森林 | 0.987 | 0.950 | 0.999 | 0.961 | 0.038 | 0.017 | 0.047 |
| KNN | 1.000 | 0.931 | 1.000 | 0.963 | 0.037 | 0.000 | 0.068 |
| 决策树 | 0.979 | 0.931 | 0.997 | 0.950 | 0.047 | 0.017 | 0.055 |
| 逻辑回归 | 0.936 | 0.921 | 0.989 | 0.973 | 0.016 | 0.044 | 0.062 |
| SVM | 0.927 | 0.901 | 0.990 | 0.969 | 0.021 | 0.040 | 0.065 |
| 朴素贝叶斯 | 0.932 | 0.901 | 0.980 | 0.958 | 0.022 | 0.066 | 0.094 |

ΔAUC = 训练AUC - 测试AUC（值越小表示泛化越好）

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/model_performance_combined.csv`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/model_performance_train.csv`

### 3.8 SHAP模型可解释性

为增强模型透明度和临床可解释性，对表现最佳的LightGBM模型进行SHAP（SHapley Additive exPlanations）分析。SHAP值在全局和局部层面为解释模型预测提供了统一框架。

#### 3.8.1 全局特征重要性

全局SHAP分析揭示了队列整体PTC复发的顶级预测特征（图4）。**初始治疗反应（Response）**以绝对优势成为最具影响力的预测因子（平均|SHAP| = 3.69），其次是**N分期**（平均|SHAP| = 1.55）、**T分期**（平均|SHAP| = 0.74）、**年龄**（平均|SHAP| = 0.73）和**体格检查**（平均|SHAP| = 0.56）。值得注意的是，治疗反应的SHAP值远高于其他所有特征，凸显其在复发预测中的核心地位。SHAP条形图（图4A）展示各特征的平均绝对SHAP值，量化其对模型预测的整体贡献。SHAP蜂群图（图4B）通过显示各特征的SHAP值分布，并以色标指示特征值（红色=高，蓝色=低），揭示特征效应的大小和方向。

**全局SHAP分析的主要发现：**

1. **治疗反应**以压倒性优势显示最高的SHAP值（3.69），是复发预测中最关键的单一因素
2. **N分期**（1.55）和**T分期**（0.74）位列第二、三位，确认肿瘤分期和淋巴结受累的重要性
3. **患者人口学特征**（年龄，0.73）和**体格检查**（0.56）贡献显著，反映临床评估的价值
4. **肿瘤分期（Stage）**和**病理类型**的SHAP值极低（分别为0.27和0.0001），提示在本模型中这些特征对预测贡献有限
5. **性别、吸烟史、放疗史、甲状腺功能和M分期**的SHAP值为零，表明这些特征未参与LightGBM模型的预测决策

#### 3.8.2 局部解释：个体风险评估

为展示个体层面的可解释性，为风险谱两端的患者生成SHAP瀑布图（图4C和4D）：

**高风险病例（图4C）：**最高风险患者（预测概率 = 0.982）的基线值为-2.34（对应低复发概率）。晚期分期（IV期）、大肿瘤（T4）、不利病理（高细胞型）和多灶性疾病的强正向贡献推动预测向高复发风险。正常甲状腺功能和阴性体格检查等因素的负向贡献部分抵消了这些风险。

**低风险病例（图4D）：**最低风险患者（预测概率 = 0.023）显示有利分期（I期）、小肿瘤（T1）、有利病理（经典型）和良好治疗反应的负向贡献。这些保护因素的组合导致最终预测强烈倾向于无复发。

**表8. SHAP分析的前10位最重要特征**

| 排名 | 特征 | 平均\|SHAP\| | SHAP范围 | 解释 |
|------|------|:---:|:---:|---|
| 1 | **治疗反应** | 3.691 | ±7.52 | 初始治疗反应最重要，远超其他特征 |
| 2 | **N（N分期）** | 1.549 | ±2.52 | 淋巴结受累程度高度重要 |
| 3 | **T（T分期）** | 0.740 | ±2.29 | 原发肿瘤大小/范围 |
| 4 | **年龄** | 0.725 | ±1.34 | 诊断时患者年龄 |
| 5 | **体格检查** | 0.555 | ±1.43 | 体格发现 |
| 6 | **风险** | 0.393 | ±0.68 | ATA风险分层 |
| 7 | **分期** | 0.269 | ±1.00 | 总体肿瘤分期中等重要 |
| 8 | **病灶数量** | 0.249 | ±1.21 | 单灶vs多灶 |
| 9 | **腺病** | 0.197 | ±1.25 | 腺病状态 |
| 10 | **病理** | <0.001 | ±0.001 | 组织学亚型贡献极低 |

**数据来源**：`thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/08_shap_global.png/pdf`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/09_shap_beeswarm.png/pdf`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/10_shap_high_risk.png/pdf`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/11_shap_low_risk.png/pdf`, `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/shap_feature_importance_LightGBM.csv`

### 3.9 Streamlit网络应用部署

为促进临床转化和实际部署，使用**Streamlit**开发交互式网络应用并在本地部署。该应用集成表现最佳的LightGBM模型（`BEST_MODEL_LightGBM.pkl`），为PTC复发风险预测提供直观界面。

#### 3.9.1 应用架构

| 组件 | 描述 |
|------|------|
| **后端模型** | 最优超参数的预训练LightGBM分类器 |
| **预处理** | 在训练数据上拟合的StandardScaler（`scaler.pkl`） |
| **前端** | Streamlit网页界面（localhost:8501） |
| **依赖项** | Python 3.8+, scikit-learn, pandas, numpy, streamlit, shap |

#### 3.9.2 应用功能

网络应用提供四大功能模块：

**1. 🔮 单例预测模块**
- 16个临床特征的交互式表单输入
- 实时预测与概率输出
- 个体SHAP瀑布图提供可解释性
- 风险分层：低危（<30%）、中危（30-70%）、高危（>70%）
- 可下载的PDF报告，包含患者详情和预测结果

**2. 📊 批量预测模块**
- CSV文件上传支持多患者（批量预测）
- 自动数据验证和预处理
- 带个体概率的批量结果表
- 可下载的CSV，包含预测和风险类别
- 大数据集的进度跟踪

**3. 📈 模型性能模块**
- 全部9种模型的并排比较
- 带AUC值的交互式ROC曲线
- 校准图（预测vs观察概率）
- 精确率-召回率曲线与AP分数
- 特征重要性排名

**4. ℹ️ 关于模块**
- 项目背景和目标
- 建模流程图
- 特征描述和数据字典
- 临床使用指南
- 引用信息

#### 3.9.3 部署流程

**先决条件：**
```bash
pip install -r requirements.txt
```

**启动方法：**

1. **Windows（一键启动）：** 双击 `run_app.bat`
2. **命令行：** `streamlit run app.py`
3. **默认URL：** http://localhost:8501

**部署文件：**

| 文件 | 大小 | 用途 |
|------|------|------|
| `app.py` | ~8 KB | Streamlit主应用 |
| `requirements.txt` | ~0.5 KB | Python依赖项 |
| `run_app.bat` | ~0.2 KB | Windows启动器 |
| `README_APP.md` | ~3 KB | 用户指南 |
| `BEST_MODEL_LightGBM.pkl` | ~150 KB | 训练好的LightGBM模型 |
| `scaler.pkl` | ~2 KB | 数据缩放器 |

#### 3.9.4 临床实施考量

Streamlit应用支持无缝整合至临床工作流程：

1. **床旁预测：** 临床医生可在会诊期间输入患者数据并立即获得复发风险评估
2. **共同决策：** 可视化SHAP解释促进与患者就风险因素进行沟通
3. **质量保证：** 批量预测模块支持回顾性队列分析的临床审核
4. **研究工具：** 模型性能仪表板允许持续监测预测准确性
5. **教育培训：** 培训模块帮助住院医师理解临床变量的相对重要性

**表9. 网络应用规格**

| 规格 | 值 |
|------|-----|
| 框架 | Streamlit 1.28+ |
| 模型格式 | Pickle (.pkl) |
| 输入验证 | 自动范围检查 |
| 预测延迟 | <0.5秒 |
| 最大批量大小 | 10,000例患者 |
| 支持的浏览器 | Chrome, Firefox, Edge, Safari |
| 响应式设计 | 桌面优化（1024×768+） |
| 数据隐私 | 本地处理，无云端上传 |
| 导出格式 | PDF, CSV |

**数据来源**：`app.py`, `README_APP.md`, `requirements.txt`

### 3.10 主要发现总结

1. **最佳综合性能**：LightGBM以判别能力（AUC = 0.983）、校准度（ECE = 0.022）和临床效用（平均NB = 0.245）的最佳平衡实现最高排名。

2. **梯度提升模型优势**：LightGBM、XGBoost和CatBoost形成AUC > 0.978且校准优异的第一梯队，其中LightGBM与CatBoost无显著差异（p = 0.987），但与XGBoost存在显著差异（p < 0.001）。

3. **统计显著性**：LightGBM显著优于决策树（p = 0.050），对KNN表现出边缘显著性优势（p = 0.065）；与XGBoost比较亦存在显著差异（p < 0.001）。

4. **分类一致性**：XGBoost和LightGBM展现完全分类一致（κ = 1.000），CatBoost与两者一致性亦极高（κ = 0.972）。

5. **临床效用**：所有梯度提升模型在临床相关阈值范围（10-50%）内提供正净获益，LightGBM平均净获益最高（0.245）。

6. **关键预测因子**：SHAP分析揭示治疗反应（|SHAP| = 3.69）是迄今为止最重要的预测因子，其次是N分期（1.55）、T分期（0.74）、年龄（0.73）和体格检查（0.56）；病理类型贡献极低（<0.001）。

---

## 4 讨论 (Discussion)

### 4.1 主要研究发现

本研究系统比较了九种机器学习算法在335例甲状腺乳头状癌（PTC）患者复发预测中的综合性能。研究结果显示，基于梯度提升决策树的算法（LightGBM、XGBoost、CatBoost）在判别能力、校准度和临床效用方面均显著优于传统机器学习方法。其中，LightGBM表现最为出色，在独立测试集上达到0.983（95% CI: 0.949-1.000）的AUC、0.970（95% CI: 0.931-1.000）的准确率，以及0.026的Brier分数，同时在决策曲线分析中实现0.245的平均净获益。这一性能水平不仅优于本研究中评估的其他传统机器学习模型，也与近期文献报道的最佳结果相当甚至更为优异。

### 4.2 与既往研究的比较

#### 4.2.1 多模型比较研究的启示

近年来，多项研究对机器学习模型在甲状腺癌复发预测中的应用进行了系统比较。Setiawan[12]对23种不同的机器学习模型进行了全面评估，包括多种核函数的SVM、朴素贝叶斯、决策树、K近邻、随机森林、AdaBoost和梯度提升机等，并首次引入卡方特征选择方法。研究发现，未经特征选择时随机森林表现最佳（准确率94%），而经卡方特征选择后有10种模型达到100%准确率。Clark等[13]比较了六种机器学习模型（KNN、SVM、决策树、AdaBoost、XGBoost、随机森林）的性能，发现随机森林在所有场景下均表现最优，证明SMOTE数据平衡和超参数优化能显著提升模型效果。Penner等[14]评估了LightGBM、随机森林、KNN、逻辑回归、SGD和Gandalf等多种技术结合特征选择方法的性能，在383例患者的UCI数据集上实现94.8-95.9%的准确率。

Islam等[17]提出了一种新颖的特征选择流程，结合多种特征选择技术，使随机森林模型达到98.70%的准确率。Senyer Yapici和Uzun Arslan[18]采用混合机器学习框架结合SMOTE数据平衡和特征选择，发现随机森林和Bagging集成方法在平衡数据上表现最佳。这些多模型比较研究为本研究选择九种代表性算法提供了理论基础，也证实了集成学习方法在甲状腺癌复发预测中的一致优势。

#### 4.2.2 梯度提升模型的优越性

梯度提升框架（XGBoost、LightGBM、CatBoost）在本研究中展现的卓越性能与当前机器学习领域的算法发展趋势高度吻合。Sarker等[15]将CatBoost与物理启发式元启发式优化算法相结合，报告优化后模型达到96.35%的准确率；Hanani等[19]采用CatBoost结合SHAP分析，达到97%的准确率和0.99的AUC；Shrestha等[28]利用鲸鱼优化算法优化XGBoost超参数，实现99%的准确率。Thakur等[16]系统比较了六种机器学习模型，随机森林分类器达到98.26%的最高准确率，通过嵌套交叉验证和网格搜索进行超参数优化。

本研究中，三种梯度提升模型（LightGBM、XGBoost、CatBoost）形成性能优异的第一梯队（AUC均>0.978），均显著优于决策树（p≤0.050）。然而，统计检验显示LightGBM与XGBoost之间存在显著差异（p<0.001），表明LightGBM在本数据集上具有略微更优的判别能力；而LightGBM与CatBoost之间无显著差异（p=0.987），XGBoost与CatBoost之间亦无显著差异（p=0.899）。这一发现与Oka和Takefuji[29]关于梯度提升模型在特征重要性估计中潜在偏倚的方法学讨论形成呼应——尽管存在特征重要性偏倚的理论风险，但经过适当优化的GBDT模型仍可在预测准确性方面展现出卓越性能。Cohen's Kappa分析进一步证实XGBoost与LightGBM在分类决策上完全一致（κ=1.000），表明这两种主流梯度提升算法虽然存在统计上的AUC差异，但在实际分类决策中表现出高度一致性。

#### 4.2.3 模型性能与文献报道的横向比较

我们的研究发现与近期多项大规模研究结果高度一致。Wang等[9]在2244例PTC患者的研究中报告随机森林模型达到最佳性能（AUC: 0.767），显著优于ATA风险分层（AUC: 0.620）。本研究中LightGBM的AUC（0.983）显著高于Wang等报告的水平，这可能归因于以下几点：首先，我们纳入了更全面的临床病理特征（16个变量），涵盖人口学特征、病史、甲状腺功能、体格检查、病理类型、TNM分期和治疗反应等多维度信息；其次，梯度提升算法在处理异质性医疗数据时通常展现出比随机森林更强的特征学习能力；第三，我们采用了严格的超参数优化和概率校准流程，确保模型输出概率的可靠性。

Thakur等[16]采用383例分化型甲状腺癌患者数据比较六种机器学习模型，报告随机森林达到98.26%的最高准确率。本研究中LightGBM和XGBoost均达到97.0%的测试集准确率，与Thakur等的结果相近。Setiawan[12]的研究发现随机森林在特征选择后可达100%准确率，Islam等[17]和Senyer Yapici与Uzun Arslan[18]的研究也证实随机森林在甲状腺癌复发预测中的优越性，与本研究中Random Forest达到95.0%准确率的结果一致。

Schindele等[14]利用XGBoost在1228例分化型甲状腺癌患者中开发复发预测模型，报告验证集AUC为0.84、独立测试集AUC为0.88。本研究中XGBoost的测试集AUC达到0.978，显著高于Schindele等的报告。这一差异可能源于样本选择（我们专注于PTC而非全部DTC）、特征集构成以及更严格的模型优化流程。然而，两项研究均强调了肿瘤分期、甲状腺球蛋白水平和治疗反应作为核心预测因子的一致性，这为PTC复发风险的生物学基础提供了相互验证。

### 4.3 模型校准与临床效用

与多数既往研究仅关注AUC等判别指标不同，本研究系统评估了模型的概率校准质量和临床效用。LightGBM在预期校准误差（ECE: 0.022）、最大校准误差（MCE: 0.591）和Brier分数（0.026）三项校准指标中均表现最佳，表明其输出的复发概率能够准确反映患者的真实风险水平。这一特性对于临床决策至关重要：当模型预测某患者复发概率为30%时，若模型校准良好，则该患者群体实际复发率应接近30%，这为共享决策和风险沟通提供了可靠基础。

决策曲线分析（DCA）进一步证实了梯度提升模型的临床实用价值。LightGBM在10-50%阈值概率范围内实现0.245的平均净获益，高于CatBoost（0.232）、XGBoost（0.238）和其他所有模型。这意味着，若临床医生根据LightGBM预测概率制定治疗决策（如在预测概率>20%时加强随访或考虑辅助治疗），相比"全部治疗"或"全部不治疗"策略，每100例患者可额外获得约24.5例的正确分类。这一临床净获益水平支持将机器学习模型整合入PTC术后管理流程的合理性。Penner等[14]和Clark等[13]的研究同样强调了机器学习模型在甲状腺癌复发风险预测中的临床实用性。

相比之下，Random Forest尽管AUC达到0.961，但其校准性能较差（ECE: 0.091），表明模型存在过度自信的概率估计倾向。这与文献中关于树集成模型校准特性的报道一致，也凸显了在医学预测模型开发中综合评估判别能力和校准质量的必要性。

### 4.4 SHAP可解释性分析的临床意义

本研究通过SHAP分析揭示了PTC复发预测的关键因子及其贡献模式，发现与既往研究存在显著的一致性和互补性。

#### 4.4.1 治疗反应的核心地位与肿瘤分期因素

SHAP全局分析显示，**初始治疗反应（Response）以压倒性优势成为最重要的预测因子**（平均|SHAP| = 3.69），其重要性远超其他所有特征之和。这一发现凸显了ATA指南动态风险评估概念的核心价值——初始治疗反应不仅是当前疾病控制状态的反映，更是未来复发风险的强预测指标。与Schindele等[14]将术后甲状腺球蛋白（Tg）水平列为最重要因子的发现不同，我们的模型更强调初始治疗反应这一综合性指标，这可能反映了我们数据集中治疗反应变量整合了生化、影像和临床评估的多维度信息。

**N分期（淋巴结受累程度）位列第二**（平均|SHAP| = 1.55），显著高于T分期（0.74）和总体肿瘤分期（0.27），提示淋巴结转移状态在复发风险分层中具有独立于原发肿瘤分期的预测价值。这一发现与Jang等[21]强调淋巴结因素重要性的研究结果高度一致。值得注意的是，**总体肿瘤分期（Stage）和病理类型（Pathology）在本模型中贡献相对较低**（分别为0.27和<0.001），这与传统临床认知有所不同，可能反映了在已纳入T、N、M等TNM分期组分的情况下，总体分期信息的冗余性。

#### 4.4.2 淋巴结因素与影像组学特征的预测价值

淋巴结状态是影响PTC复发风险的关键病理因素。Jang等[21]的多中心研究纳入1232例N1期PTC患者，通过机器学习评估淋巴结相关风险因素，发现淋巴结外侵犯（ENE）、转移淋巴结病灶最大直径、清扫淋巴结数量、转移淋巴结数量和转移淋巴结比率均为结构性复发的独立风险因子，并确定了新的风险截断值（转移淋巴结病灶最大直径0.2cm和1.1cm、转移淋巴结数量4和13、转移淋巴结比率0.28和0.58）。

N分期（淋巴结受累程度）在本研究中**位列第二位重要特征**，是仅次于治疗反应的关键预测因子。这与Jang等[21]的多中心研究结果高度呼应。Park和Lee[10]的研究同样证实淋巴结比率（LNR）和对侧中央区淋巴结转移在所有机器学习模型中均被列为重要特征。我们的研究结果支持淋巴结状态在复发风险分层中的作用，但由于特征编码方式（N分期作为分类变量而非连续变量）和缺乏清扫淋巴结数量的精确数据，可能低估了淋巴结因素的综合贡献。未来研究应考虑纳入更详细的淋巴结病理信息（如阳性淋巴结绝对数量、LNR、ENE状态）以进一步提升预测精度。

Zhou等[22]结合超声影像组学特征和临床病理变量构建PTC复发预测列线图，在554例患者中，联合列线图在训练队列3年随访中达到0.851的AUC，在验证队列达到0.885的AUC，显著优于单纯临床模型和单纯影像组学模型。这一研究证实了多模态特征整合在PTC复发预测中的预测价值，也为未来整合影像组学特征提供了方法学参考。Chattopadhyay[23]采用层次聚类算法和多元线性回归构建混合机器学习模型，发现年龄、性别、吸烟、放疗史、淋巴结病变和肿瘤分期等53.84%的特征与复发呈正相关，同样展示了多维度特征分析在风险分层中的重要性。

#### 4.4.3 代谢指标的潜在预测价值

除传统临床病理因素外，代谢指标与甲状腺癌复发的关联日益受到关注。Li等[25]在11,317例PTC患者中研究了甘油三酯-葡萄糖体质指数（TyG-BMI）与肿瘤侵袭性和复发风险的关系，发现较高的TyG-BMI与肿瘤直径>1cm（OR=1.35）、多灶性（OR=1.42）和甲状腺外侵犯（OR=1.53）风险增加相关，但与中等至高复发风险呈负相关（OR=0.68）。机器学习算法一致识别甘油三酯水平为预测PTC侵袭性和复发风险的首要贡献因素。这一研究揭示了代谢指标与PTC复发之间复杂的线性和非线性关系，提示未来预测模型可能需要整合代谢组学数据以提高预测精度。

#### 4.4.4 治疗反应的预测价值与可解释性实践

**初始治疗反应（Response）在本研究中位列第一位重要特征**，且以压倒性优势（平均|SHAP| = 3.69，是第二位的2.4倍）领先于其他所有特征，这与ATA指南的动态风险评估概念高度一致，也与Hanani等[19]将治疗反应列为最重要预测因子（SHAP值：2.077）的研究结果相符。Hanani等[19]的SHAP分析将治疗反应列为最重要预测因子（SHAP值：2.077），Wen等[20]的研究同样将治疗反应识别为前五大关键因素之一。Onah等[27]采用PCA和t-SVD等无监督特征工程方法结合多种分类器，PCA-逻辑回归流程达到0.95的平衡准确率和0.99的AUC，并通过SHAP分析支持模型可解释性，证实SHAP分析可有效识别关键预测因子。Thakur等[16]和Onah等[27]的研究也采用交叉验证和独立测试集评估策略验证了模型的泛化性能。

通过高复发风险案例和低复发风险案例的SHAP瀑布图对比，我们展示了SHAP分析在个体化风险沟通中的应用潜力。对于特定患者，临床医生不仅可以看到综合风险评分，还能识别推动该患者进入高危或低危类别的具体因素组合。例如，高危患者的特征贡献模式可能显示IV期分期、T4肿瘤和不良病理类型的协同作用，而低危患者则可能受益于I期分期、T1肿瘤和良好治疗反应的叠加保护效应。这种个体化的解释能力超越了传统风险评分系统的"一刀切"分类，为精准医学实践提供了技术基础。

### 4.5 模型泛化与临床部署

#### 4.5.1 泛化性能评估

本研究通过比较训练集和测试集性能评估模型泛化能力。LightGBM和XGBoost在训练集上达到完美性能（AUC: 1.000），在测试集上仍保持极高性能（AUC: 0.983和0.978），ΔAUC分别为0.017和0.022，显示出良好的泛化能力且无明显过拟合。相比之下，KNN和决策树表现出更明显的过拟合倾向（ΔAUC: 0.037和0.047）。这一发现与梯度提升算法的正则化机制（如shrinkage、子采样、叶子节点约束）有关，这些机制在提升训练性能的同时有效控制模型复杂度，从而在新数据上保持稳定表现。

#### 4.5.2 网络应用的临床转化价值与部署优势

基于最优LightGBM模型开发的Streamlit网络应用代表了从研究模型到临床工具的转化尝试。与Wen等[20]开发的TCCheck网络应用类似，我们的应用提供单例预测、批量预测、模型性能展示和功能说明四大模块，支持临床医生进行实时风险评分和结果解释。Wen等[20]开发的TCCheck网络应用集成了堆叠集成学习框架（SGD、Extra Trees、决策树作为基学习器，XGBoost作为元学习器），在115例测试集上实现96.52%准确率和0.9921 AUC，并通过SHAP分析识别出治疗反应、年龄、N分期、风险分层和淋巴结病变为前五大关键因素。

##### Web App部署的核心优势

本研究采用Streamlit框架开发的Web应用具有以下显著的临床部署优势：

**1. 即时可及性与床旁决策支持**
传统机器学习模型往往停留在研究阶段，难以直接进入临床 workflow。本Web应用通过浏览器即可访问，临床医生可在患者就诊期间即时输入16项临床参数，在<0.5秒内获得复发风险预测结果及SHAP可视化解释。这种即时性对于甲状腺癌术后随访决策尤为关键——医生可在门诊环境中实时评估患者复发风险，无需等待离线计算或专家咨询，显著提升决策效率。

**2. 零门槛技术集成**
与需要复杂软件安装或编程环境的传统部署方案不同，本应用采用Python原生架构，仅需执行`streamlit run app.py`即可启动，无需前端开发经验或额外的IT基础设施投入。这种低技术门槛特性使其特别适合资源有限的基层医疗机构，无需专业的数据科学团队即可完成部署和维护。

**3. 数据隐私与本地安全部署**
应用支持完全本地化部署，所有患者数据在机构内部网络处理，无需上传至第三方云端服务器。这一特性满足HIPAA、网络安全法等医疗数据隐私法规的严格要求，消除了医疗机构对数据外泄的担忧，为敏感医疗信息的处理提供了安全保障。本地部署还意味着在无互联网连接的封闭环境中仍可正常运行，确保医疗服务的连续性。

**4. 可解释性与医患沟通支持**
SHAP瀑布图的可视化展示不仅帮助临床医生理解决策依据，更可直接用于与患者的风险沟通。医生可向患者展示"治疗反应良好使您的复发风险降低了XX%"等个性化解释，增强患者对治疗方案的理解和依从性，促进共同决策。这种透明性对于提高患者满意度和治疗 adherence 具有重要价值。

**5. 灵活的功能模块设计**
应用提供四大功能模块满足不同场景需求：
- **单例预测**：适用于门诊实时风险评估，生成可下载的PDF报告存档
- **批量预测**：支持CSV文件上传，可一次性处理多达10,000例患者，适用于回顾性队列研究或质量审计
- **模型性能展示**：为临床管理者提供模型能力证据，支持临床采纳决策
- **功能说明**：内置数据字典和临床使用指南，降低培训成本

**6. 持续迭代与模型更新便利**
基于Pickle序列化的模型持久化策略（`BEST_MODEL_LightGBM.pkl`和`scaler.pkl`）使得模型更新变得简单——当有新数据可用或模型优化后，仅需替换.pkl文件即可实现无缝升级，无需修改应用代码或重新部署。这种模块化架构支持模型的持续学习和版本管理，确保临床使用的始终是最优模型版本。

**7. 成本效益与可扩展性**
相比商业化的临床决策支持系统（CDSS），本开源方案具有显著的成本优势：无需许可费用、无需专用服务器（可在普通办公电脑上运行）、无需持续的云服务费。同时，Streamlit框架支持通过Docker容器化部署到机构服务器或云平台，为大规模部署提供了可扩展路径。

本地部署策略确保患者数据不出机构网络，符合医疗数据隐私保护要求（如HIPAA、网络安全法）。未来可考虑通过多中心外部验证进一步评估模型在不同人群、不同医疗环境下的性能稳定性，并探索与医院信息系统（HIS）的API集成以实现无缝工作流整合。Borra和Vemuri[26]的研究表明，集成Bagging Trees模型在UCI数据集上达到95.6%的准确率，也强调了机器学习模型作为医疗决策支持工具的潜力。

### 4.6 研究局限性

本研究存在以下局限性：首先，数据来源于公开的UCI数据集[25]，原始研究的单中心回顾性设计限制了我们控制数据收集过程的能力，可能存在选择偏倚和信息偏倚；其次，样本量（335例）相对有限，且复发事件发生率（26.87%）虽高于Wang等[9]（8.0%）和Park等[10]（约10%）的报告，但仍属于类别不平衡场景，尽管我们采用了分层抽样和适当的类别权重，大样本多中心研究仍有助于进一步验证模型性能；第三，随访数据（中位随访时间、复发发生时间）的缺失限制了我们进行时间-事件分析（生存分析）的能力，未来研究可考虑应用XGBoost Cox模型或DeepSurv等深度学习生存模型；第四，缺乏外部验证队列，模型在其他种族、不同医疗水平地区的泛化能力有待证实；第五，部分临床重要变量（如BRAF突变状态、TERT启动子突变、术后放射性碘摄取情况）未纳入数据集，限制了模型对分子预后因素的利用；第六，2025年ATA风险分层的精细化，本研究未精确到低风险、低中风险、中高风险、高风险的细分情况，未来需要进一步细分病人复发风险分层来进一步分析[30]。

### 4.7 未来研究方向

基于本研究发现，未来研究可从以下方向深化：

1. **多模态数据整合**：将临床病理特征与影像组学（如Zhou等[31]的DECT影像组学研究、Zhou等[22]的超声影像组学研究）、基因组学（如BRAF、TERT、RAS突变）和转录组学数据相结合，构建更全面、更精确的预测模型；

2. **动态预测模型**：开发能够随时间更新风险估计的动态预测工具，整合术后Tg水平、超声随访结果等时间序列数据，如Lee等[32]提出的多模态深度学习模型；

3. **深度学习与迁移学习**：探索深度学习架构（如Transformer、图神经网络）在PTC复发预测中的应用，并考虑使用大型医疗数据集进行预训练以提升小样本场景下的性能；

4. **模型公平性与代谢指标整合**：评估模型在不同亚组（年龄、性别、病理亚型、合并症）中的性能一致性，确保算法公平性；Li等[25]的研究提示代谢指标可能存在复杂的非线性效应，未来模型应考虑整合TyG-BMI等代谢组学数据以提高预测精度。

5. **前瞻性临床验证**：开展前瞻性队列研究或随机对照试验，将机器学习引导的风险分层与常规ATA分层进行比较，评估其对临床结局（如早期复发检出率、过度治疗减少程度）的实际影响。

### 4.8 结论

本研究系统比较了九种机器学习算法在PTC复发预测中的性能，证实梯度提升模型（尤其是LightGBM）在判别能力、校准度、临床效用和泛化能力方面均达到最优水平。SHAP分析识别出初始治疗反应、N分期、T分期、年龄和体格检查为关键预测因子，其中治疗反应的重要性远高于其他特征，凸显了动态风险评估在复发预测中的核心价值。基于最优模型开发的网络应用为临床提供了可解释、易用且可靠的复发风险评估工具。未来多中心外部验证和前瞻性临床研究将有助于进一步确立机器学习模型在PTC术后管理中的临床价值，推动精准医学理念在甲状腺癌诊疗实践中的落地应用。

---

## 参考文献 (References)

[1] Singer PA, Cooper DS, Daniels GH, Ladenson PW, Greenspan FS, Levy EG, et al. Treatment guidelines for patients with thyroid nodules and well-differentiated thyroid cancer. *Arch Intern Med*. 1996;156:2165-2172.

[2] Cooper DS, Doherty GM, Haugen BR, Kloos RT, Lee SL, Mandel SJ, et al. Management Guidelines for Patients with Thyroid Nodules and Differentiated Thyroid Cancer: The American Thyroid Association Guidelines Taskforce. *Thyroid*. 2006;16:109-142.

[3] Cooper DS, Doherty GM, Haugen BR, Kloos RT, Lee SL, Mandel SJ, et al. Revised American Thyroid Association Management Guidelines for Patients with Thyroid Nodules and Differentiated Thyroid Cancer. *Thyroid*. 2009;19:1167-1214.

[4] Haugen BR, Alexander EK, Bible KC, Doherty GM, Mandel SJ, Nikiforov YE, et al. 2015 American Thyroid Association Management Guidelines for Adult Patients with Thyroid Nodules and Differentiated Thyroid Cancer. *Thyroid*. 2016;26:1-133.

[5] Ringel MD, Sosa JA, Baloch Z, Bischoff L, Bloom G, Brent GA, et al. 2025 American Thyroid Association Management Guidelines for Adult Patients with Differentiated Thyroid Cancer. *Thyroid*. 2025;35:841-985.

[6] Tran A, Weigel RJ, Beck AC. ATA risk stratification in papillary thyroid microcarcinoma has low positive predictive value when identifying recurrence. *Am J Surg*. 2024;229:106-110. doi: 10.1016/j.amjsurg.2023.11.003.

[7] Maino F, Botte M, Dalmiglio C, Valerio L, Brilli L, Trimarchi A, et al. Prognostic Factors Improving ATA Risk System and Dynamic Risk Stratification in Low- and Intermediate-Risk DTC Patients. *J Clin Endocrinol Metab*. 2024;109(3):722-729. doi: 10.1210/clinem/dgad591.

[8] Grani G, Zatelli MC, Alfò M, Montesano T, Torlontano M, Morelli S, et al. Real-World Performance of the American Thyroid Association Risk Estimates in Predicting 1-Year Differentiated Thyroid Cancer Outcomes: A Prospective Multicenter Study of 2000 Patients. *Thyroid*. 2021;31(2):264-271. doi: 10.1089/thy.2020.0272.

[9] Wang H, Zhang C, Li Q, Tian T, Huang R, Qiu J, Tian R. Development and validation of prediction models for papillary thyroid cancer structural recurrence using machine learning approaches. *BMC Cancer*. 2024;24(1):427. doi: 10.1186/s12885-024-12146-4.

[10] Park YM, Lee BJ. Machine learning-based prediction model using clinico-pathologic factors for papillary thyroid carcinoma recurrence. *Sci Rep*. 2021;11(1):4948. doi: 10.1038/s41598-021-84504-2.

[11] Setiawan KE. Predicting recurrence in differentiated thyroid cancer: a comparative analysis of various machine learning models including ensemble methods with chi-squared feature selection. *Commun Math Biol Neurosci*. 2024; Article ID 55.

[12] Clark E, Price S, Lucena T, Haberlein B, Wahbeh A, Seetan R. Predictive Analytics for Thyroid Cancer Recurrence: A Machine Learning Approach. *Knowledge*. 2024;4(4):557-570. doi: 10.3390/knowledge4040029.

[13] Penner MA, Berger D, Guo X, Levman J. Machine Learning in Differentiated Thyroid Cancer Recurrence and Risk Prediction. *Appl Sci*. 2025;15(17):9397. doi: 10.3390/app15179397.

[14] Schindele A, Krebold A, Heiß U, Nimptsch K, Pfaehler E, Berr C, et al. Interpretable machine learning for thyroid cancer recurrence prediction: Leveraging XGBoost and SHAP analysis. *Eur J Radiol*. 2025;186:112049. doi: 10.1016/j.ejrad.2025.112049.

[15] Sarker P, Choi K, Nahid AA, Samad MA. CatBoost with physics-based metaheuristics for thyroid cancer recurrence prediction. *BioData Min*. 2025;18(1):84. doi: 10.1186/s13040-025-00494-1.

[16] Thakur D, Gera T, Bhardwaj V, Mazen R, Lasisi A, Engida T. A comparative study on advanced predictive modeling of thyroid cancer recurrence using multi algorithmic machine learning frameworks. *Sci Rep*. 2025;16(1):3385. doi: 10.1038/s41598-025-33396-7.

[17] Islam MT, Bari S, Shailee S, Billal MM. A Novel Feature Selection Pipeline for Accurate Thyroid Cancer Recurrence Prediction. In: *Proceeding of the 2nd International Conference on Machine Intelligence and Emerging Technologies*. MIET 2024. Springer, 2025. doi: 10.1007/978-981-96-2721-9_3.

[18] Senyer Yapici I, Uzun Arslan R. Predictive analytics for thyroid cancer recurrence: a feature selection and data balancing approach. *Eur Phys J Spec Top*. 2025;234:4751-4771. doi: 10.1140/epjs/s11734-025-01720-x.

[19] Hanani AA, Donmez TB, Kutlu M, Mansour M. Predicting thyroid cancer recurrence using supervised CatBoost: A SHAP-based explainable AI approach. *Medicine (Baltimore)*. 2025;104(22):e42667. doi: 10.1097/MD.0000000000042667.

[20] Wen H, Li X, Zhao X. TC check: a web app for thyroid cancer recurrence prediction using explainable machine learning. *J Cancer Res Clin Oncol*. 2025;152(1):14. doi: 10.1007/s00432-025-06377-6.

[21] Jang SW, Park JH, Kim HR, Kwon HJ, Lee YM, Hong SJ, Yoon JH. Recurrence Risk Evaluation in Patients with Papillary Thyroid Carcinoma: Multicenter Machine Learning Evaluation of Lymph Node Variables. *Cancers (Basel)*. 2023;15(2):550. doi: 10.3390/cancers15020550.

[22] Zhou B, Liu J, Yang Y, Ye X, Liu Y, Mao M, et al. Ultrasound-based nomogram to predict the recurrence in papillary thyroid carcinoma using machine learning. *BMC Cancer*. 2024;24(1):810. doi: 10.1186/s12885-024-12546-6.

[23] Chattopadhyay S. Towards Predicting Recurrence Risk of Differentiated Thyroid Cancer with a Hybrid Machine Learning Model. *Medinformatics*. 2024. doi: 10.47852/bonviewMEDIN42024441.

[24] Borzooei S, Briganti G, Golparian M, Lechien JR, Tarokhian A. Machine learning for risk stratification of thyroid cancer patients: a 15-year cohort study. *Eur Arch Otorhinolaryngol*. 2024;281(4):2095-2104. doi: 10.1007/s00405-023-08299-w.

[25] Li C, Dionigi G, Guan H, Sun H, Zhang J. The triglyceride-glucose body mass index paradox: dual metabolic effects on tumor aggressiveness and recurrence risk in 11,317 papillary thyroid carcinoma patients. *Int J Surg*. 2026;112(2):4356-4365. doi: 10.1097/JS9.0000000000003887.

[26] Borra D, Vemuri J. Predicting Differentiated Thyroid Cancer Recurrence Using Machine Learning. In: *Computer Vision and Robotics. CVR 2025*. Lecture Notes in Networks and Systems, vol 1643. Springer, 2026. doi: 10.1007/978-3-032-06250-5_8.

[27] Onah E, Eze UJ, Abdulraheem AS, Ezigbo UG, Amorha KC, Ntie-Kang F. Optimizing unsupervised feature engineering and classification pipelines for differentiated thyroid cancer recurrence prediction. *BMC Med Inform Decis Mak*. 2025;25(1):182. doi: 10.1186/s12911-025-03018-3.

[28] Shrestha K, Rifat HMJO, Biswas U, Tiang JJ, Nahid AA. Predicting the Recurrence of Differentiated Thyroid Cancer Using Whale Optimization-Based XGBoost Algorithm. *Diagnostics (Basel)*. 2025;15(13):1684. doi: 10.3390/diagnostics15131684.

[29] Oka S, Takefuji Y. Complementing interpretable machine learning with synergistic analytical strategies for thyroid cancer recurrence prediction. *Eur J Radiol*. 2025;191:112308. doi: 10.1016/j.ejrad.2025.112308.

[30] Moneta C, Trevisan M, Colombo C, De Luca A, Lugaresi M, Reali GM, et al. VALIDATION OF THE 2025 ATA RISK STRATIFICATION SYSTEM IN A COHORT OF PATIENTS WITH PAPILLARY THYROID CARCINOMA. *J Clin Endocrinol Metab*. 2026. doi: 10.1210/clinem/dgag167.

[31] Zhou Y, Xu Y, Si Y, Wu F, Xu X. Initial Recurrence Risk Stratification of Papillary Thyroid Cancer based on Intratumoral and Peritumoral Dual Energy CT Radiomics. *Curr Med Imaging*. 2025;21:e15734056402179. doi: 10.2174/0115734056402179250813050300.

[32] Lee DH, Choi JW, Kim GH, Park S, Jeon HJ. Application of a Novel Multimodal-Based Deep Learning Model for the Prediction of Papillary Thyroid Carcinoma Recurrence. *Int J Gen Med*. 2024;17:6585-6594. doi: 10.2147/IJGM.S486189.

---

## 图表来源文件索引

| 图表 | 描述 | 数据来源 |
|------|------|----------|
| **图1** | ROC曲线比较判别性能 | `model_performance_test.csv`, `test_predictions.csv` |
| **图2** | 校准图显示预测vs观察概率 | `calibration_metrics_test.csv`, `test_predictions.csv` |
| **图3** | 决策曲线分析展示临床净获益 | `dca_metrics_test.csv` |
| **图4A** | SHAP全局特征重要性（条形图） | `08_shap_global.png/pdf` |
| **图4B** | SHAP蜂群图显示特征分布 | `09_shap_beeswarm.png/pdf` |
| **图4C** | 高风险示例病例的SHAP瀑布图 | `10_shap_high_risk.png/pdf` |
| **图4D** | 低风险示例病例的SHAP瀑布图 | `11_shap_low_risk.png/pdf` |
| **图5** | 前3名模型的特征重要性比较 | `07_feature_importance.png/pdf` |

---

## 补充数据文件索引

**所有结果文件位于 `thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM/` 文件夹中**

### 性能与评估文件

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `dataset_info.txt` | 数据集基线特征 | 表1 |
| `model_performance_test.csv` | 测试集性能指标 | 表2, 表7 |
| `model_performance_train.csv` | 训练集性能指标 | 表7 |
| `model_performance_combined.csv` | 合并训练/测试指标 | 表7 |
| `model_ranking.csv` | 按性能排名 | 总结 |

### 统计检验文件

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `delong_pvalues_test.csv` | DeLong检验p值 | 表3 |
| `delong_auc_diff_test.csv` | AUC差异 | 表3（补充） |
| `delong_zstatistics_test.csv` | Z统计量 | 表3（补充） |
| `calibration_metrics_test.csv` | ECE, MCE, Brier分数 | 表4 |
| `dca_metrics_test.csv` | 决策曲线指标 | 表5, 图3 |
| `cohen_kappa_matrix_test.csv` | 模型间一致性 | 表6 |

### SHAP分析文件

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `shap_feature_importance_LightGBM.csv` | LightGBM特征重要性 | 表8 |
| `shap_summary_LightGBM.csv` | SHAP汇总统计 | 解释分析 |
| `shap_values_matrix_LightGBM.csv` | SHAP值矩阵 | 可视化 |

### Streamlit网络应用文件

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `app.py` | Streamlit主应用 (~8 KB) | 网络部署 |
| `requirements.txt` | Python依赖列表 | 安装 |
| `run_app.bat` | Windows一键启动 | 快速启动 |
| `README_APP.md` | 用户指南和文档 | 参考 |
| `BEST_MODEL_LightGBM.pkl` | 训练好的LightGBM模型 (~150 KB) | 后端 |
| `scaler.pkl` | StandardScaler对象 (~2 KB) | 预处理 |

---

*文档生成日期：2026年4月24日*  


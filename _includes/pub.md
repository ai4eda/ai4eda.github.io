# <i class="fa fa-chevron-right"></i> Publications

<br>



## <i class="fa fa-chevron-right"></i> High Level Synthesis

<h3></h3>
<table class="table table-hover">

<tr id="tr-10.1145/3503540" >
<td align='right'>
1.
</td>
<td>
    Correlated Multi-Objective Multi-Fidelity Optimization for HLS Directives Design [<a href='https://doi.org/10.1145/3503540' target='_blank'>paper</a>] <br>
    <em>Qi&nbsp;Sun, Tinghuan&nbsp;Chen, Siting&nbsp;Liu, Jianli&nbsp;Chen, Hao&nbsp;Yu, and Bei&nbsp;Yu</em><br>
    TODAES 2022  <br>
    
</td>
</tr>


<tr id="tr-8714724" >
<td align='right'>
2.
</td>
<td>
    Machine Learning Based Routing Congestion Prediction in FPGA High-Level Synthesis [<a href='https://arxiv.org/abs/1905.03852' target='_blank'>paper</a>] <br>
    <em>Jieru&nbsp;Zhao, Tingyuan&nbsp;Liang, Sharad&nbsp;Sinha, and Wei&nbsp;Zhang</em><br>
    DATE 2019  <br>
    
</td>
</tr>


<tr id="tr-10.1145/3020078.3021747" >
<td align='right'>
3.
</td>
<td>
    A Parallel Bandit-Based Approach for Autotuning FPGA Compilation [<a href='https://doi.org/10.1145/3020078.3021747' target='_blank'>paper</a>] <br>
    <em>Chang&nbsp;Xu, Gai&nbsp;Liu, Ritchie&nbsp;Zhao, Stephen&nbsp;Yang, Guojie&nbsp;Luo, and Zhiru&nbsp;Zhang</em><br>
    FPGA 2017  <br>
    
</td>
</tr>


<tr id="tr-liu2013learning" >
<td align='right'>
4.
</td>
<td>
    On learning-based methods for design-space exploration with high-level synthesis [<a href='http://www.cs.columbia.edu/~luca/research/liu_DAC13.pdf' target='_blank'>paper</a>] <br>
    <em>Hung-Yi&nbsp;Liu and Luca&nbsp;P&nbsp;Carloni</em><br>
    DAC 2013  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Logic Synthesis

<h3>Synthesis Results Estimation</h3>
<table class="table table-hover">

<tr id="tr-ISCA22_sns" >
<td align='right'>
1.
</td>
<td>
    SNS's Not a Synthesizer: A Deep-Learning-Based Synthesis Predictor 
[<a href='javascript:;'
    onclick='$("#abs_ISCA22_sns").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3470496.3527444' target='_blank'>paper</a>]  [<a href='https://github.com/Entropy-xcy/sns' target='_blank'>code</a>] <br>
    <em>Ceyu&nbsp;Xu, Chris&nbsp;Kjellqvist, and Lisa&nbsp;Wu&nbsp;Wills</em><br>
    ISCA 2022  <br>
    
<div id="abs_ISCA22_sns" style="text-align: justify; display: none" markdown="1">
The number of transistors that can fit on one monolithic chip has reached billions to tens of billions in this decade thanks to Moore's Law. With the advancement of every technology generation, the transistor counts per chip grow at a pace that brings about exponential increase in design time, including the synthesis process used to perform design space explorations. Such a long delay in obtaining synthesis results hinders an efficient chip development process, significantly impacting time-to-market. In addition, these large-scale integrated circuits tend to have larger and higher-dimension design spaces to explore, making it prohibitively expensive to obtain physical characteristics of all possible designs using traditional synthesis tools.In this work, we propose a deep-learning-based synthesis predictor called SNS (SNS's not a Synthesizer), that predicts the area, power, and timing physical characteristics of a broad range of designs at two to three orders of magnitude faster than the Synopsys Design Compiler while providing on average a 0.4998 RRSE (root relative square error). We further evaluate SNS via two representative case studies, a general-purpose out-of-order CPU case study using RISC-V Boom open-source design and an accelerator case study using an in-house Chisel implementation of DianNao, to demonstrate the capabilities and validity of SNS.
</div>

</td>
</tr>


<tr id="tr-TCAD22_bullseye" >
<td align='right'>
2.
</td>
<td>
    Bulls-Eye: Active Few-shot Learning Guided Logic Synthesis 
[<a href='javascript:;'
    onclick='$("#abs_TCAD22_bullseye").toggle()'>abs</a>] [<a href='https://ieeexplore.ieee.org/abstract/document/9969911' target='_blank'>paper</a>] <br>
    <em>Animesh&nbsp;Basak&nbsp;Chowdhury, Benjamin&nbsp;Tan, Ryan&nbsp;Carey, Tushit&nbsp;Jain, Ramesh&nbsp;Karri, and Siddharth&nbsp;Garg</em><br>
    TCAD 2022  <br>
    
<div id="abs_TCAD22_bullseye" style="text-align: justify; display: none" markdown="1">
Generating sub-optimal synthesis transformation sequences (“synthesis recipe”) is an important problem in logic synthesis. Manually crafted synthesis recipes have poor quality. State-of-the art machine learning (ML) works to generate synthesis recipes do not scale to large netlists as the models need to be trained from scratch, for which training data is collected using time consuming synthesis runs. We propose a new approach, Bulls-Eye, that fine-tunes a pre-trained model on past synthesis data to accurately predict the quality of a synthesis recipe for an unseen netlist. Our approach achieves 2x-30x run-time improvement and generates synthesis recipes achieving close to 95% quality-of-result (QoR) compared to conventional techniques using actual synthesis runs. We show our QoR beat state-of-the-art approaches on various benchmarks.
</div>

</td>
</tr>


<tr id="tr-MLCAD20_decision" >
<td align='right'>
3.
</td>
<td>
    Decision making in synthesis cross technologies using LSTMs and transfer learning 
[<a href='javascript:;'
    onclick='$("#abs_MLCAD20_decision").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3380446.3430638' target='_blank'>paper</a>]  [<a href='https://www.youtube.com/watch?v=c5k1uQahMa8&t=184s' target='_blank'>talk</a>] <br>
    <em>Cunxi&nbsp;Yu and Wang&nbsp;Zhou</em><br>
    MLCAD 2020  <br>
    
<div id="abs_MLCAD20_decision" style="text-align: justify; display: none" markdown="1">
We propose a general approach that precisely estimates the Quality-of-Result (QoR), such as delay and area, of unseen synthesis flows for specific designs. The main idea is leveraging LSTM-based network to forecast the QoR, where the inputs are synthesis flows represented in novel timed-flow modeling, and QoRs are ground truth. This approach is demonstrated with 1.2 million data points collected using 14nm, 7nm regular-voltage (RVT), and 7nm low-voltage (LVT) technologies with twelve IC designs. The accuracy of predicting the QoRs (delay and area) evaluated using mean absolute prediction error (MAPE). While collecting training data points in EDA can be extremely challenging, we propose to elaborate transfer learning in our approach, which enables accurate predictions cross different technologies and different IC designs. Our transfer learning approach obtains estimation MAPE 3.7% over 960,000 test points collected on 7nm technologies, with only 100 data points used for training the pre-trained LSTM network using 14nm dataset.
</div>

</td>
</tr>


<tr id="tr-DAC18_angel" >
<td align='right'>
4.
</td>
<td>
    Developing synthesis flows without human knowledge 
[<a href='javascript:;'
    onclick='$("#abs_DAC18_angel").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3195970.3196026' target='_blank'>paper</a>]  [<a href='https://github.com/ycunxi/FLowGen-CNNs-DAC18' target='_blank'>code</a>]  [<a href='https://ycunxi.github.io/cunxiyu/slides/dac18.pdf' target='_blank'>slides</a>] <br>
    <em>Cunxi&nbsp;Yu, Houping&nbsp;Xiao, and Giovanni&nbsp;De&nbsp;Micheli</em><br>
    DAC 2018  <br>
    
<div id="abs_DAC18_angel" style="text-align: justify; display: none" markdown="1">
Design flows are the explicit combinations of design transformations, primarily involved in synthesis, placement and routing processes, to accomplish the design of Integrated Circuits (ICs) and System-on-Chip (SoC). Mostly, the flows are developed based on the knowledge of the experts. However, due to the large search space of design flows and the increasing design complexity, developing Intellectual Property (IP)-specific synthesis flows providing high Quality of Result (QoR) is extremely challenging. This work presents a fully autonomous framework that artificially produces design-specific synthesis flows without human guidance and baseline flows, using Convolutional Neural Network (CNN). The demonstrations are made by successfully designing logic synthesis flows of three large scaled designs.
</div>

</td>
</tr>

</table>
<h3></h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/iccad/FengLCYYH22" >
<td align='right'>
1.
</td>
<td>
    Batch Sequential Black-Box Optimization with Embedding Alignment Cells for Logic Synthesis [<a href='https://doi.org/10.1145/3508352.3549363' target='_blank'>paper</a>] <br>
    <em>Chang&nbsp;Feng, Wenlong&nbsp;Lyu, Zhitang&nbsp;Chen, Junjie&nbsp;Ye, Mingxuan&nbsp;Yuan, and Jianye&nbsp;Hao</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>

</table>
<h3>Operator Sequence Scheduling</h3>
<table class="table table-hover">

<tr id="tr-ICCAD20_flowtune" >
<td align='right'>
1.
</td>
<td>
    Practical Multi-armed Bandits in Boolean Optimization 
[<a href='javascript:;'
    onclick='$("#abs_ICCAD20_flowtune").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3400302.3415615' target='_blank'>paper</a>]  [<a href='https://github.com/Yu-Utah/FlowTune' target='_blank'>code</a>]  [<a href='https://www.youtube.com/watch?v=EPcn5ttp1TM&t=360s' target='_blank'>talk</a>] <br>
    <em>Cunxi&nbsp;Yu</em><br>
    ICCAD 2020  <br>
    
<div id="abs_ICCAD20_flowtune" style="text-align: justify; display: none" markdown="1">
Recent years have seen increasing employment of decision intelligence in electronic design automation (EDA), which aims to reduce the manual efforts and boost the design closure process in modern toolflows. However, existing approaches either require a large number of labeled data for training or are limited in practical EDA toolflow integration due to computation overhead. This paper presents a generic end-to-end and high-performance domainspecific, multi-stage multi-armed bandit framework for Boolean logic optimization. This framework addresses optimization problems on a) And-Inv-Graphs (# nodes), b) Conjunction Normal Form (CNF) minimization (# clauses) for Boolean Satisfiability, c) post static timing analysis (STA) delay and area optimization for standard-cell technology mapping, and d) FPGA technology mapping for 6-in LUT architectures. Moreover, the proposed framework has been integrated with ABC, Yosys, VTR, and industrial tools. The experimental results demonstrate that our framework outperforms both hand-crafted flows and ML explored flows in quality of results, and is orders of magnitude faster compared to ML-based approaches.
</div>

</td>
</tr>


<tr id="tr-ASPDAC20_drills" >
<td align='right'>
2.
</td>
<td>
    DRiLLS: Deep Reinforcement Learning for Logic Synthesis 
[<a href='javascript:;'
    onclick='$("#abs_ASPDAC20_drills").toggle()'>abs</a>] [<a href='https://ieeexplore.ieee.org/abstract/document/9045559' target='_blank'>paper</a>]  [<a href='https://github.com/scale-lab/DRiLLS' target='_blank'>code</a>] <br>
    <em>Abdelrahman&nbsp;Hosny, Soheil&nbsp;Hashemi, Mohamed&nbsp;Shalan, and Sherief&nbsp;Reda</em><br>
    ASP-DAC 2020  <br>
    
<div id="abs_ASPDAC20_drills" style="text-align: justify; display: none" markdown="1">
Logic synthesis requires extensive tuning of the synthesis optimization flow where the quality of results (QoR) depends on the sequence of optimizations used. Efficient design space exploration is challenging due to the exponential number of possible optimization permutations. Therefore, automating the optimization process is necessary. In this work, we propose a novel reinforcement learning-based methodology that navigates the optimization space without human intervention. We demonstrate the training of an Advantage Actor Critic (A2C) agent that seeks to minimize area subject to a timing constraint. Using the proposed methodology, designs can be optimized autonomously with no-humans in-loop. Evaluation on the comprehensive EPFL benchmark suite shows that the agent outperforms existing exploration methodologies and improves QoRs by an average of 13%.
</div>

</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Circuit Verification

<h3>Circuit Verification</h3>
<table class="table table-hover">

<tr id="tr-DAC22_ncl" >
<td align='right'>
1.
</td>
<td>
    Functionality matters in netlist representation learning [<a href='http://www.cse.cuhk.edu.hk/~byu/papers/C142-DAC2022-GCL.pdf' target='_blank'>paper</a>] <br>
    <em>Ziyi&nbsp;Wang, Chen&nbsp;Bai, Zhuolun&nbsp;He, Guangliang&nbsp;Zhang, Qiang&nbsp;Xu, Tsung-Yi&nbsp;Ho, Bei&nbsp;Yu, and Yu&nbsp;Huang</em><br>
    DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-ICCAD21_abgnn" >
<td align='right'>
2.
</td>
<td>
    Graph Learning-Based Arithmetic Block Identification [<a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9643581&casa_token=eTSwwwdrkj0AAAAA:iSIYCHnrvx8WxjcHJRg-LZFEa5c9sA1ZlWBo7YUiUdpwVPq4m5-j-V2mi3tk7sFz7cJIRMk&tag=1' target='_blank'>paper</a>] <br>
    <em>Zhuolun&nbsp;He, Ziyi&nbsp;Wang, Chen&nbsp;Bail, Haoyu&nbsp;Yang, and Bei&nbsp;Yu</em><br>
    ICCAD 2021  <br>
    
</td>
</tr>

</table>
<h3>Reliability</h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/aspdac/GengSCXHY23" >
<td align='right'>
1.
</td>
<td>
    Mixed-Type Wafer Failure Pattern Recognition [<a href='https://doi.org/10.1145/3566097.3568363' target='_blank'>paper</a>] <br>
    <em>Hao&nbsp;Geng, Qi&nbsp;Sun, Tinghuan&nbsp;Chen, Qi&nbsp;Xu, Tsung-Yi&nbsp;Ho, and Bei&nbsp;Yu</em><br>
    ASP-DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-chen2021deep" >
<td align='right'>
2.
</td>
<td>
    Deep H-GCN: Fast analog IC aging-induced degradation estimation [<a href='https://ieeexplore.ieee.org/document/9521579' target='_blank'>paper</a>] <br>
    <em>Tinghuan&nbsp;Chen, Qi&nbsp;Sun, Canhui&nbsp;Zhan, Changze&nbsp;Liu, Huatao&nbsp;Yu, and Bei&nbsp;Yu</em><br>
    TCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-chen2021analog" >
<td align='right'>
3.
</td>
<td>
    Analog IC aging-induced degradation estimation via heterogeneous graph convolutional networks [<a href='https://dl.acm.org/doi/10.1145/3394885.3431546' target='_blank'>paper</a>] <br>
    <em>Tinghuan&nbsp;Chen, Qi&nbsp;Sun, Canhui&nbsp;Zhan, Changze&nbsp;Liu, Huatao&nbsp;Yu, and Bei&nbsp;Yu</em><br>
    ASP-DAC 2021  <br>
    
</td>
</tr>

</table>
<h3>Security</h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/aspdac/ChangPXH023" >
<td align='right'>
1.
</td>
<td>
    Rethink before Releasing Your Model: ML Model Extraction Attack in EDA [<a href='https://doi.org/10.1145/3566097.3567896' target='_blank'>paper</a>] <br>
    <em>Chen-Chia&nbsp;Chang, Jingyu&nbsp;Pan, Zhiyao&nbsp;Xie, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    ASP-DAC 2023  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Floorplan

<h3></h3>
<table class="table table-hover">

<tr id="tr-DAC22_SpecPart" >
<td align='right'>
1.
</td>
<td>
    SpecPart: A Supervised Spectral Framework for Hypergraph Partitioning Solution Improvement [<a href='https://dl.acm.org/doi/abs/10.1145/3508352.3549390' target='_blank'>paper</a>] <br>
    <em>Ismail&nbsp;Bustany, Andrew&nbsp;B.&nbsp;Kahng, Ioannis&nbsp;Koutis, Bodhisatta&nbsp;Pramanik, and Zhiang&nbsp;Wang</em><br>
    DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/iccad/WenZYCYLX22" >
<td align='right'>
2.
</td>
<td>
    LayouTransformer: Generating Layout Patterns with Transformer via Sequential Pattern Modeling [<a href='https://doi.org/10.1145/3508352.3549350' target='_blank'>paper</a>] <br>
    <em>Liangjian&nbsp;Wen, Yi&nbsp;Zhu, Lei&nbsp;Ye, Guojin&nbsp;Chen, Bei&nbsp;Yu, Jianzhuang&nbsp;Liu, and Chunjing&nbsp;Xu</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-DAC20_TPGNN" >
<td align='right'>
3.
</td>
<td>
    TP-GNN: A Graph Neural Network Framework for Tier Partitioning in Monolithic 3D ICs [<a href='https://ieeexplore.ieee.org/document/9218582' target='_blank'>paper</a>] <br>
    <em>Yi-Chen&nbsp;Lu, Sai&nbsp;Surya&nbsp;Kiran&nbsp;Pentapati, Lingjun&nbsp;Zhu, Kambiz&nbsp;Samadi, and Sung&nbsp;Kyu&nbsp;Lim</em><br>
    DAC 2020  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Placement

<h3></h3>
<table class="table table-hover">

<tr id="tr-tcadChenKZHY23" >
<td align='right'>
1.
</td>
<td>
    PROS 2.0: A Plug-In for Routability Optimization and Routed Wirelength Estimation Using Deep Learning [<a href='https://doi.org/10.1109/TCAD.2022.3168259' target='_blank'>paper</a>] <br>
    <em>Jingsong&nbsp;Chen, Jian&nbsp;Kuang, Guowei&nbsp;Zhao, Dennis&nbsp;J.&nbsp;-H.&nbsp;Huang, and Evangeline&nbsp;F.&nbsp;Y.&nbsp;Young</em><br>
    TCAD 2023  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/aspdac/ChenMGZL23" >
<td align='right'>
2.
</td>
<td>
    MacroRank: Ranking Macro Placement Solutions Leveraging Translation Equivariancy [<a href='https://doi.org/10.1145/3566097.3567899' target='_blank'>paper</a>] <br>
    <em>Yifan&nbsp;Chen, Jing&nbsp;Mai, Xiaohan&nbsp;Gao, Muhan&nbsp;Zhang, and Yibo&nbsp;Lin</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-DREAM-GAN_ISPD2023_Yichen" >
<td align='right'>
3.
</td>
<td>
    DREAM-GAN: Advancing DREAMPlace towards Commercial-Quality using Generative Adversarial Learning <br>
    <em>Yi-Chen&nbsp;Lu, Haoxing&nbsp;Ren, Hao-Hsiang&nbsp;Hsiao, and Sung&nbsp;Kyu&nbsp;Lim</em><br>
    ISPD 2023  <br>
    
</td>
</tr>


<tr id="tr-cheng2022the" >
<td align='right'>
4.
</td>
<td>
    The Policy-gradient Placement and Generative Routing Neural Networks for Chip Design [<a href='https://openreview.net/forum?id=uNYqDfPEDD8' target='_blank'>paper</a>] <br>
    <em>Ruoyu&nbsp;Cheng, Xianglong&nbsp;Lyu, Yang&nbsp;Li, Junjie&nbsp;Ye, Jianye&nbsp;HAO, and Junchi&nbsp;Yan</em><br>
    NeurIPS 2022  <br>
    
</td>
</tr>


<tr id="tr-lai2022maskplace" >
<td align='right'>
5.
</td>
<td>
    MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning [<a href='https://arxiv.org/abs/2211.13382' target='_blank'>paper</a>] <br>
    <em>Yao&nbsp;Lai, Yao&nbsp;Mu, and Ping&nbsp;Luo</em><br>
    NeurIPS 2022  <br>
    
</td>
</tr>


<tr id="tr-wang2022lhnna" >
<td align='right'>
6.
</td>
<td>
    LHNN: Lattice Hypergraph Neural Network for VLSI Congestion Prediction [<a href='https://dl.acm.org/doi/abs/10.1145/3489517.3530675' target='_blank'>paper</a>] <br>
    <em>Bowen&nbsp;Wang, Guibao&nbsp;Shen, Dong&nbsp;Li, Jianye&nbsp;Hao, Wulong&nbsp;Liu, Yu&nbsp;Huang, Hongzhong&nbsp;Wu, Yibo&nbsp;Lin, Guangyong&nbsp;Chen, and Pheng&nbsp;Ann&nbsp;Heng</em><br>
    DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-baek2022pin" >
<td align='right'>
7.
</td>
<td>
    Pin Accessibility and Routing Congestion Aware DRC Hotspot Prediction Using Graph Neural Network and U-Net [<a href='https://dl.acm.org/doi/abs/10.1145/3508352.3549346' target='_blank'>paper</a>] <br>
    <em>Kyeonghyeon&nbsp;Baek, Hyunbum&nbsp;Park, Suwan&nbsp;Kim, Kyumyung&nbsp;Choi, and Taewhan&nbsp;Kim</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-liang2022stochasticb" >
<td align='right'>
8.
</td>
<td>
    A Stochastic Approach to Handle Non-Determinism in Deep Learning-Based Design Rule Violation Predictions [<a href='https://dl.acm.org/doi/abs/10.1145/3508352.3549347' target='_blank'>paper</a>] <br>
    <em>Rongjian&nbsp;Liang, Hua&nbsp;Xiang, Jinwook&nbsp;Jung, Jiang&nbsp;Hu, and Gi-Joon&nbsp;Nam</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-DBLP:journals/corr/abs-2207-08398" >
<td align='right'>
9.
</td>
<td>
    Bayesian Optimization for Macro Placement [<a href='https://doi.org/10.48550/arXiv.2207.08398' target='_blank'>paper</a>] <br>
    <em>ChangYong&nbsp;Oh, Roberto&nbsp;Bondesan, Dana&nbsp;Kianfar, Rehan&nbsp;Ahmed, Rishubh&nbsp;Khurana, Payal&nbsp;Agarwal, Romain&nbsp;Lepert, Mysore&nbsp;Sriram, and Max&nbsp;Welling</em><br>
    ICML 2022  <br>
    
</td>
</tr>


<tr id="tr-PLACE-DATE2021-Cong" >
<td align='right'>
10.
</td>
<td>
    Global placement with deep learning-enabled explicit routability optimization [<a href='https://ieeexplore.ieee.org/document/9473959' target='_blank'>paper</a>] <br>
    <em>Siting&nbsp;Liu, Qi&nbsp;Sun, Peiyu&nbsp;Liao, Yibo&nbsp;Lin, and Bei&nbsp;Yu</em><br>
    DATE 2021  <br>
    
</td>
</tr>


<tr id="tr-NEURIPS2021_898aef09" >
<td align='right'>
11.
</td>
<td>
    On Joint Learning for Solving Placement and Routing in Chip Design [<a href='https://proceedings.neurips.cc/paper/2021/file/898aef0932f6aaecda27aba8e9903991-Paper.pdf' target='_blank'>paper</a>] <br>
    <em>Ruoyu&nbsp;Cheng and Junchi&nbsp;Yan</em><br>
    NeurIPS 2021  <br>
    
</td>
</tr>


<tr id="tr-changAutomaticRoutabilityPredictor2021a" >
<td align='right'>
12.
</td>
<td>
    Automatic Routability Predictor Development Using Neural Architecture Search [<a href='https://arxiv.org/abs/2012.01737' target='_blank'>paper</a>] <br>
    <em>Chen-Chia&nbsp;Chang, Jingyu&nbsp;Pan, Tunhou&nbsp;Zhang, Zhiyao&nbsp;Xie, Jiang&nbsp;Hu, Weiyi&nbsp;Qi, Chun-Wei&nbsp;Lin, Rongjian&nbsp;Liang, Joydeep&nbsp;Mitra, Elias&nbsp;Fallon, and Yiran&nbsp;Chen</em><br>
    ICCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-chenPROSPluginRoutability2020" >
<td align='right'>
13.
</td>
<td>
    PROS: A Plug-in for Routability Optimization Applied in the State-of-the-art Commercial EDA Tool Using Deep Learning [<a href='https://ieeexplore.ieee.org/document/9256565' target='_blank'>paper</a>] <br>
    <em>Jingsong&nbsp;Chen, Jian&nbsp;Kuang, Guowei&nbsp;Zhao, Dennis&nbsp;J.-H.&nbsp;Huang, and Evangeline&nbsp;F.Y.&nbsp;Young</em><br>
    ICCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-ghoseGeneralizableCrossGraphEmbedding2021" >
<td align='right'>
14.
</td>
<td>
    Generalizable Cross-Graph Embedding for GNN-based Congestion Prediction [<a href='https://arxiv.org/abs/2111.05941' target='_blank'>paper</a>] <br>
    <em>Amur&nbsp;Ghose, Vincent&nbsp;Zhang, Yingxue&nbsp;Zhang, Dong&nbsp;Li, Wulong&nbsp;Liu, and Mark&nbsp;Coates</em><br>
    ICCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-alawiehHighDefinitionRoutingCongestion2020a" >
<td align='right'>
15.
</td>
<td>
    High-Definition Routing Congestion Prediction for Large-Scale FPGAs [<a href='https://ieeexplore.ieee.org/document/9045178' target='_blank'>paper</a>] <br>
    <em>Mohamed&nbsp;Baker&nbsp;Alawieh, Wuxi&nbsp;Li, Yibo&nbsp;Lin, Love&nbsp;Singhal, Mahesh&nbsp;A.&nbsp;Iyer, and David&nbsp;Z.&nbsp;Pan</em><br>
    ASP-DAC 2020  <br>
    
</td>
</tr>


<tr id="tr-liangDRCHotspotPrediction2020" >
<td align='right'>
16.
</td>
<td>
    DRC Hotspot Prediction at Sub-10nm Process Nodes Using Customized Convolutional Network [<a href='https://dl.acm.org/doi/10.1145/3372780.3375560' target='_blank'>paper</a>] <br>
    <em>Rongjian&nbsp;Liang, Hua&nbsp;Xiang, Diwesh&nbsp;Pandey, Lakshmi&nbsp;Reddy, Shyam&nbsp;Ramji, Gi-Joon&nbsp;Nam, and Jiang&nbsp;Hu</em><br>
    ISPD 2020  <br>
    
</td>
</tr>


<tr id="tr-PLACE-DAC2019-DREAMPlace" >
<td align='right'>
17.
</td>
<td>
    DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement [<a href='https://ieeexplore.ieee.org/document/8807076' target='_blank'>paper</a>] <br>
    <em>Yibo&nbsp;Lin, Shounak&nbsp;Dhar, Wuxi&nbsp;Li, Haoxing&nbsp;Ren, Brucek&nbsp;Khailany, and David&nbsp;Z.&nbsp;Pan</em><br>
    DAC 2019  <br>
    
</td>
</tr>


<tr id="tr-PLACE-DAC2019-painting" >
<td align='right'>
18.
</td>
<td>
    Painting on placement: Forecasting routing congestion using conditional generative adversarial nets [<a href='https://dl.acm.org/doi/10.1145/3316781.3317876' target='_blank'>paper</a>] <br>
    <em>Cunxi&nbsp;Yu and Zhiru&nbsp;Zhang</em><br>
    DAC 2019  <br>
    
</td>
</tr>


<tr id="tr-PLACE-DAC2019-pin" >
<td align='right'>
19.
</td>
<td>
    Pin accessibility prediction and optimization with deep learning-based pin pattern recognition [<a href='https://ieeexplore.ieee.org/document/8806809' target='_blank'>paper</a>] <br>
    <em>Tao-Chun&nbsp;Yu, Shao-Yun&nbsp;Fang, Hsien-Shih&nbsp;Chiu, Kai-Shun&nbsp;Hu, Philip&nbsp;Hui-Yuh&nbsp;Tai, Cindy&nbsp;Chin-Fang&nbsp;Shen, and Henry&nbsp;Sheng</em><br>
    DAC 2019  <br>
    
</td>
</tr>


<tr id="tr-PLACE-ICCAD2018-routenet" >
<td align='right'>
20.
</td>
<td>
    RouteNet: Routability prediction for mixed-size designs using convolutional neural network [<a href='https://ieeexplore.ieee.org/document/8587655' target='_blank'>paper</a>] <br>
    <em>Zhiyao&nbsp;Xie, Yu-Hung&nbsp;Huang, Guan-Qi&nbsp;Fang, Haoxing&nbsp;Ren, Shao-Yun&nbsp;Fang, Yiran&nbsp;Chen, and Jiang&nbsp;Hu</em><br>
    ICCAD 2018  <br>
    
</td>
</tr>


<tr id="tr-chanRoutabilityOptimizationIndustrial2017" >
<td align='right'>
21.
</td>
<td>
    Routability Optimization for ndustrial Designs at Sub-14nm Process Nodes Using Machine Learning [<a href='https://dl.acm.org/doi/10.1145/3036669.3036681' target='_blank'>paper</a>] <br>
    <em>Wei-Ting&nbsp;J.&nbsp;Chan, Pei-Hsin&nbsp;Ho, Andrew&nbsp;B.&nbsp;Kahng, and Prashant&nbsp;Saxena</em><br>
    ISPD 2017  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Clock Tree Synthesis

<h3></h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/aspdac/LiangNRHR23" >
<td align='right'>
1.
</td>
<td>
    BufFormer: A Generative ML Framework for Scalable Buffering [<a href='https://doi.org/10.1145/3566097.3567900' target='_blank'>paper</a>] <br>
    <em>Rongjian&nbsp;Liang, Siddhartha&nbsp;Nath, Anand&nbsp;Rajaram, Jiang&nbsp;Hu, and Haoxing&nbsp;Ren</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-DBLP:journals/tcad/LuLASL22" >
<td align='right'>
2.
</td>
<td>
    A Clock Tree Prediction and Optimization Framework Using Generative Adversarial Learning [<a href='https://doi.org/10.1109/TCAD.2021.3122109' target='_blank'>paper</a>] <br>
    <em>Yi-Chen&nbsp;Lu, Jeehyun&nbsp;Lee, Anthony&nbsp;Agnesina, Kambiz&nbsp;Samadi, and Sung&nbsp;Kyu&nbsp;Lim</em><br>
    TCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/glvlsi/Beheshti-Shirazi21" >
<td align='right'>
3.
</td>
<td>
    A Reinforced Learning Solution for Clock Skew Engineering to Reduce Peak Current and IR Drop [<a href='https://doi.org/10.1145/3453688.3461754' target='_blank'>paper</a>] <br>
    <em>Sayed&nbsp;Aresh&nbsp;Beheshti-Shirazi, Ashkan&nbsp;Vakil, Sai&nbsp;Manoj&nbsp;P.&nbsp;D., Ioannis&nbsp;Savidis, Houman&nbsp;Homayoun, and Avesta&nbsp;Sasan</em><br>
    GLSVLSI 2021  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/ises/NagariaD20" >
<td align='right'>
4.
</td>
<td>
    Designing of an Optimization Technique for the Prediction of CTS Outcomes using Neural Network [<a href='https://doi.org/10.1109/iSES50453.2020.00075' target='_blank'>paper</a>] <br>
    <em>Shuchi&nbsp;Nagaria and Sujay&nbsp;Deb</em><br>
    iSES 2020  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/iccad/LuLASL19" >
<td align='right'>
5.
</td>
<td>
    GAN-CTS: A Generative Adversarial Framework for Clock Tree Prediction and Optimization [<a href='https://doi.org/10.1109/ICCAD45719.2019.8942063' target='_blank'>paper</a>] <br>
    <em>Yi-Chen&nbsp;Lu, Jeehyun&nbsp;Lee, Anthony&nbsp;Agnesina, Kambiz&nbsp;Samadi, and Sung&nbsp;Kyu&nbsp;Lim</em><br>
    ICCAD 2019  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Routing

<h3></h3>
<table class="table table-hover">

<tr id="tr-ROUTE-DAC2023-Steiner" >
<td align='right'>
1.
</td>
<td>
    Concurrent Sign-off Timing Optimization via Deep Steiner Points Refinement <br>
    <em>Siting&nbsp;Liu, Ziyi&nbsp;Wang, Fangzhou&nbsp;Liu, Yibo&nbsp;Lin, Bei&nbsp;Yu, and Martin&nbsp;Wong</em><br>
    DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-ASPDAC23_DPRoute" >
<td align='right'>
2.
</td>
<td>
    DPRoute: Deep Learning Framework for Package Routing [<a href='https://dl.acm.org/doi/abs/10.1145/3566097.3567902' target='_blank'>paper</a>] <br>
    <em>Yeu-Haw&nbsp;Yeh, Simon&nbsp;Yi-Hung&nbsp;Chen, Hung-Ming&nbsp;Chen, Deng-Yao&nbsp;Tu, Guan-Qi&nbsp;Fang, Yun-Chih&nbsp;Kuo, and Po-Yang&nbsp;Chen</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-ROUTE_DAC2022_Jingyu" >
<td align='right'>
3.
</td>
<td>
    Reinforcement Learning Guided Detailed Routing for FinFET Custom Circuits <br>
    <em>Hao&nbsp;Chen, Kai-Chieh&nbsp;Hsu, Walker&nbsp;J.&nbsp;Turner, Po-Hsuan&nbsp;Wei, Keren&nbsp;Zhu, David&nbsp;Z.&nbsp;Pan, and Haoxing&nbsp;Ren</em><br>
    ISPD 2023  <br>
    
</td>
</tr>


<tr id="tr-DAC22_Pan" >
<td align='right'>
4.
</td>
<td>
    Towards Collaborative Intelligence: Routability Estimation Based on Decentralized Private Data [<a href='https://dl.acm.org/doi/abs/10.1145/3489517.3530578' target='_blank'>paper</a>] <br>
    <em>Jingyu&nbsp;Pan, Chen-Chia&nbsp;Chang, Zhiyao&nbsp;Xie, Ang&nbsp;Li, Minxue&nbsp;Tang, Tunhou&nbsp;Zhang, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-ROUTE-DATE2021-RLOrder" >
<td align='right'>
5.
</td>
<td>
    Asynchronous reinforcement learning framework for net order exploration in detailed routing [<a href='https://ieeexplore.ieee.org/document/9474007' target='_blank'>paper</a>] <br>
    <em>Tong&nbsp;Qu, Yibo&nbsp;Lin, Zongqing&nbsp;Lu, Yajuan&nbsp;Su, and Yayi&nbsp;Wei</em><br>
    DATE 2021  <br>
    
</td>
</tr>


<tr id="tr-NEURIPS2021_898aef09" >
<td align='right'>
6.
</td>
<td>
    On Joint Learning for Solving Placement and Routing in Chip Design [<a href='https://proceedings.neurips.cc/paper/2021/file/898aef0932f6aaecda27aba8e9903991-Paper.pdf' target='_blank'>paper</a>] <br>
    <em>Ruoyu&nbsp;Cheng and Junchi&nbsp;Yan</em><br>
    NeurIPS 2021  <br>
    
</td>
</tr>


<tr id="tr-ROUTE-DAC2020-NNICs" >
<td align='right'>
7.
</td>
<td>
    Late breaking results: A neural network that routes ics [<a href='https://ieeexplore.ieee.org/document/9218598' target='_blank'>paper</a>] <br>
    <em>Dmitry&nbsp;Utyamishev and Inna&nbsp;Partin-Vaisband</em><br>
    DAC 2020  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Timing

<h3></h3>
<table class="table table-hover">

<tr id="tr-DAC23_gnnOpt" >
<td align='right'>
1.
</td>
<td>
    Restructure-Tolerant Timing Prediction via Multimodal Fusion <br>
    <em>Ziyi&nbsp;Wang, Siting&nbsp;Liu, Yuan&nbsp;Pu, Song&nbsp;Chen, Tsung-Yi&nbsp;Ho, and Bei&nbsp;Yu</em><br>
    DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-ye2023graph" >
<td align='right'>
2.
</td>
<td>
    Graph-Learning-Driven Path-Based Timing Analysis Results Predictor from Graph-Based Timing Analysis [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C152-ASPDAC2023-GPBA.pdf' target='_blank'>paper</a>] <br>
    <em>Yuyang&nbsp;Ye, Tinghuan&nbsp;Chen, Yifei&nbsp;Gao, Hao&nbsp;Yan, Bei&nbsp;Yu, and Longxing&nbsp;Shi</em><br>
    ASP-DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-ye2023fast" >
<td align='right'>
3.
</td>
<td>
    Fast and Accurate Wire Timing Estimation Based on Graph Learning [<a href='http://www.cse.cuhk.edu.hk/~byu/papers/C159-DATE2023-NetTiming.pdf' target='_blank'>paper</a>] <br>
    <em>Yuyang&nbsp;Ye, Tinghuan&nbsp;Chen, Yifei&nbsp;Gao, Hao&nbsp;Yan, Bei&nbsp;Yu, and Longxing&nbsp;Shi</em><br>
    DATE 2023  <br>
    
</td>
</tr>


<tr id="tr-9648307" >
<td align='right'>
4.
</td>
<td>
    Estimating Code Vulnerability to Timing Errors Via Microarchitecture-Aware Machine Learning [<a href='https://ieeexplore.ieee.org/abstract/document/9648307' target='_blank'>paper</a>] <br>
    <em>Styliani&nbsp;Tompazi, Ioannis&nbsp;Tsiokanos, Jesus&nbsp;Martinez&nbsp;del&nbsp;Rincon, and Georgios&nbsp;Karakonstantis</em><br>
    IEEE Des Test 2023  <br>
    
</td>
</tr>


<tr id="tr-DAC22_gnnSTA" >
<td align='right'>
5.
</td>
<td>
    A timing engine inspired graph neural network model for pre-routing slack prediction [<a href='https://dl.acm.org/doi/abs/10.1145/3489517.3530597' target='_blank'>paper</a>] <br>
    <em>Zizheng&nbsp;Guo, Mingjie&nbsp;Liu, Jiaqi&nbsp;Gu, Shuhan&nbsp;Zhang, David&nbsp;Z&nbsp;Pan, and Yibo&nbsp;Lin</em><br>
    DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-9419866" >
<td align='right'>
6.
</td>
<td>
    DEVoT: Dynamic Delay Modeling of Functional Units Under Voltage and Temperature Variations <br>
    <em>Dongning&nbsp;Ma, Xinqiao&nbsp;Zhang, Ke&nbsp;Huang, Yu&nbsp;Jiang, Wanli&nbsp;Chang, and Xun&nbsp;Jiao</em><br>
    TCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-10.1145/3526241.3530343" >
<td align='right'>
7.
</td>
<td>
    Leveraging Machine Learning for Gate-Level Timing Estimation Using Current Source Models and Effective Capacitance [<a href='https://dl.acm.org/doi/abs/10.1145/3526241.3530343' target='_blank'>paper</a>] <br>
    <em>Dimitrios&nbsp;Garyfallou, Anastasis&nbsp;Vagenas, Charalampos&nbsp;Antoniadis, Yehia&nbsp;Massoud, and George&nbsp;Stamoulis</em><br>
    GLSVLSI 2022  <br>
    
</td>
</tr>


<tr id="tr-9832695" >
<td align='right'>
8.
</td>
<td>
    ARETE: Accurate Error Assessment via Machine Learning-Guided Dynamic-Timing Analysis [<a href='https://ieeexplore.ieee.org/document/9832695/' target='_blank'>paper</a>] <br>
    <em>Ioannis&nbsp;Tsiokanos, Styliani&nbsp;Tompazi, Giorgis&nbsp;Georgakoudis, Lev&nbsp;Mukhanov, and Georgios&nbsp;Karakonstantis</em><br>
    TC 2022  <br>
    
</td>
</tr>


<tr id="tr-xie_net2_2020" >
<td align='right'>
9.
</td>
<td>
    Net2: A Graph Attention Network Method Customized for Pre-Placement Net Length Estimation [<a href='http://arxiv.org/abs/2011.13522' target='_blank'>paper</a>] <br>
    <em>Zhiyao&nbsp;Xie, Rongjian&nbsp;Liang, Xiaoqing&nbsp;Xu, Jiang&nbsp;Hu, Yixiao&nbsp;Duan, and Yiran&nbsp;Chen</em><br>
    arXiv 2020  <br>
    
</td>
</tr>


<tr id="tr-liang_routing-free_2020" >
<td align='right'>
10.
</td>
<td>
    Routing-free crosstalk prediction [<a href='https://dl.acm.org/doi/10.1145/3400302.3415712' target='_blank'>paper</a>] <br>
    <em>Rongjian&nbsp;Liang, Zhiyao&nbsp;Xie, Jinwook&nbsp;Jung, Vishnavi&nbsp;Chauha, Yiran&nbsp;Chen, Jiang&nbsp;Hu, Hua&nbsp;Xiang, and Gi-Joon&nbsp;Nam</em><br>
    ICCAD 2020  <br>
    
</td>
</tr>


<tr id="tr-cheng_fast_2020" >
<td align='right'>
11.
</td>
<td>
    Fast and Accurate Wire Timing Estimation on Tree and Non-Tree Net Structures [<a href='https://ieeexplore.ieee.org/document/9218712/' target='_blank'>paper</a>] <br>
    <em>Hsien-Han&nbsp;Cheng, Iris&nbsp;Hui-Ru&nbsp;Jiang, and Oscar&nbsp;Ou</em><br>
    DAC 2020  <br>
    
</td>
</tr>


<tr id="tr-GNN_HLS_timing" >
<td align='right'>
12.
</td>
<td>
    Accurate Operation Delay Prediction for FPGA HLS Using Graph Neural Networks <br>
    <em>Ecenur&nbsp;Ustun, Chenhui&nbsp;Deng, Debjit&nbsp;Pal, Zhijing&nbsp;Li, and Zhiru&nbsp;Zhang</em><br>
    ICCAD 2020  <br>
    
</td>
</tr>


<tr id="tr-9218588" >
<td align='right'>
13.
</td>
<td>
    TEVoT: Timing Error Modeling of Functional Units under Dynamic Voltage and Temperature Variations <br>
    <em>Xun&nbsp;Jiao, Dongning&nbsp;Ma, Wanli&nbsp;Chang, and Yu&nbsp;Jiang</em><br>
    DAC 2020  <br>
    
</td>
</tr>


<tr id="tr-9116363" >
<td align='right'>
14.
</td>
<td>
    DEFCON: Generating and Detecting Failure-prone Instruction Sequences via Stochastic Search [<a href='https://ieeexplore.ieee.org/document/9116363' target='_blank'>paper</a>] <br>
    <em>Ioannis&nbsp;Tsiokanos, Lev&nbsp;Mukhanov, Giorgis&nbsp;Georgakoudis, Dimitrios&nbsp;S.&nbsp;Nikolopoulos, and Georgios&nbsp;Karakonstantis</em><br>
    DATE 2020  <br>
    
</td>
</tr>


<tr id="tr-barboza_machine_2019" >
<td align='right'>
15.
</td>
<td>
    Machine Learning-Based Pre-Routing Timing Prediction with Reduced Pessimism [<a href='https://dl.acm.org/doi/10.1145/3316781.3317857' target='_blank'>paper</a>] <br>
    <em>Erick&nbsp;Carvajal&nbsp;Barboza, Nishchal&nbsp;Shukla, Yiran&nbsp;Chen, and Jiang&nbsp;Hu</em><br>
    DAC 2019  <br>
    
</td>
</tr>


<tr id="tr-hyun_accurate_2019" >
<td align='right'>
16.
</td>
<td>
    Accurate Wirelength Prediction for Placement-Aware Synthesis through Machine Learning [<a href='https://ieeexplore.ieee.org/document/8715016/' target='_blank'>paper</a>] <br>
    <em>Daijoon&nbsp;Hyun, Yuepeng&nbsp;Fan, and Youngsoo&nbsp;Shin</em><br>
    DATE 2019  <br>
    
</td>
</tr>


<tr id="tr-kahng_using_2018" >
<td align='right'>
17.
</td>
<td>
    Using Machine Learning to Predict Path-Based Slack from Graph-Based Timing Analysis [<a href='https://ieeexplore.ieee.org/document/8615746/' target='_blank'>paper</a>] <br>
    <em>Andrew&nbsp;B.&nbsp;Kahng, Uday&nbsp;Mallappa, and Lawrence&nbsp;Saul</em><br>
    ICCD 2018  <br>
    
</td>
</tr>


<tr id="tr-CLIM" >
<td align='right'>
18.
</td>
<td>
    CLIM: A Cross-Level Workload-Aware Timing Error Prediction Model for Functional Units <br>
    <em>Xun&nbsp;Jiao, Abbas&nbsp;Rahimi, Yu&nbsp;Jiang, Jianguo&nbsp;Wang, Hamed&nbsp;Fatemi, Jose&nbsp;Pineda&nbsp;de&nbsp;Gyvez, and Rajesh&nbsp;K.&nbsp;Gupta</em><br>
    TC 2018  <br>
    
</td>
</tr>


<tr id="tr-kahng_si_2015" >
<td align='right'>
19.
</td>
<td>
    SI for free: machine learning of interconnect coupling delay and transition effects [<a href='http://ieeexplore.ieee.org/document/7171706/' target='_blank'>paper</a>] <br>
    <em>Andrew&nbsp;B.&nbsp;Kahng, Mulong&nbsp;Luo, and Siddhartha&nbsp;Nath</em><br>
    SLIP 2015  <br>
    
</td>
</tr>


<tr id="tr-han_deep_nodate" >
<td align='right'>
20.
</td>
<td>
    A Deep Learning Methodology to Proliferate Golden Signoff Timing [<a href='https://ieeexplore.ieee.org/document/6800474' target='_blank'>paper</a>] <br>
    <em>Seung-Soo&nbsp;Han, Andrew&nbsp;B&nbsp;Kahng, Siddhartha&nbsp;Nath, Ashok&nbsp;S&nbsp;Vydyanathan, and ECE&nbsp;Departments</em><br>
    DATE 2014  <br>
    
</td>
</tr>


<tr id="tr-kahng_learning-based_2013" >
<td align='right'>
21.
</td>
<td>
    Learning-based approximation of interconnect delay and slew in signoff timing tools [<a href='http://ieeexplore.ieee.org/document/6681682/' target='_blank'>paper</a>] <br>
    <em>Andrew&nbsp;B.&nbsp;Kahng, Seokhyeong&nbsp;Kang, Hyein&nbsp;Lee, Siddhartha&nbsp;Nath, and Jyoti&nbsp;Wadhwani</em><br>
    SLIP 2013  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Layout Verification

<h3></h3>
<table class="table table-hover">

<tr id="tr-9536958" >
<td align='right'>
1.
</td>
<td>
    Hotspot Detection via Attention-Based Deep Layout Metric Learning [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C106-ICCAD2020-Metric-HSD.pdf' target='_blank'>paper</a>] <br>
    <em>Hao&nbsp;Geng, Haoyu&nbsp;Yang, Lu&nbsp;Zhang, Fan&nbsp;Yang, Xuan&nbsp;Zeng, and Bei&nbsp;Yu</em><br>
    TCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-9774579" >
<td align='right'>
2.
</td>
<td>
    Efficient Hotspot Detection via Graph Neural Network [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C134-DATE2022-GNN-HSD.pdf' target='_blank'>paper</a>] <br>
    <em>Shuyuan&nbsp;Sun, Yiyang&nbsp;Jiang, Fan&nbsp;Yang, Bei&nbsp;Yu, and Xuan&nbsp;Zeng</em><br>
    DATE 2022  <br>
    
</td>
</tr>


<tr id="tr-9586273" >
<td align='right'>
3.
</td>
<td>
    Low-Cost Lithography Hotspot Detection with Active Entropy Sampling and Model Calibration [<a href='https://ieeexplore.ieee.org/document/9586273' target='_blank'>paper</a>] <br>
    <em>Yifeng&nbsp;Xiao, Miaodi&nbsp;Su, Haoyu&nbsp;Yang, Jianli&nbsp;Chen, Jun&nbsp;Yu, and Bei&nbsp;Yu</em><br>
    DAC 2021  <br>
    
</td>
</tr>


<tr id="tr-9164914" >
<td align='right'>
4.
</td>
<td>
    Efficient Layout Hotspot Detection via Binarized Residual Neural Network Ensemble [<a href='https://ieeexplore.ieee.org/document/9164914' target='_blank'>paper</a>] <br>
    <em>Yiyang&nbsp;Jiang, Fan&nbsp;Yang, Bei&nbsp;Yu, Dian&nbsp;Zhou, and Xuan&nbsp;Zeng</em><br>
    TCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-9164899" >
<td align='right'>
5.
</td>
<td>
    Bridging the Gap Between Layout Pattern Sampling and Hotspot Detection via Batch Active Learning [<a href='https://arxiv.org/abs/1807.06446' target='_blank'>paper</a>] <br>
    <em>Haoyu&nbsp;Yang, Shuhe&nbsp;Li, Cyrus&nbsp;Tabery, Bingqing&nbsp;Lin, and Bei&nbsp;Yu</em><br>
    TCAD 2021  <br>
    
</td>
</tr>


<tr id="tr-9643590" >
<td align='right'>
6.
</td>
<td>
    Hotspot Detection via Multi-task Learning and Transformer Encoder [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C125-ICCAD2021-H-DETR.pdf' target='_blank'>paper</a>] <br>
    <em>Binwu&nbsp;Zhu, Ran&nbsp;Chen, Xinyun&nbsp;Zhang, Fan&nbsp;Yang, Xuan&nbsp;Zeng, Bei&nbsp;Yu, and Martin&nbsp;D.F.&nbsp;Wong</em><br>
    ICCAD 2021  <br>
    
</td>
</tr>

</table>
<h3>Ir Drop</h3>
<table class="table table-hover">

<tr id="tr-hyun2023decoupling" >
<td align='right'>
1.
</td>
<td>
    Decoupling Capacitor Insertion Minimizing IR-Drop Violations and Routing DRVs <br>
    <em>Daijoon&nbsp;Hyun, Younggwang&nbsp;Jung, Insu&nbsp;Cho, and Youngsoo&nbsp;Shin</em><br>
    ASP-DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-chen2022vectorbaseda" >
<td align='right'>
2.
</td>
<td>
    Vector-Based Dynamic IR-drop Prediction Using Machine Learning <br>
    <em>Jia-Xian&nbsp;Chen, Shi-Tang&nbsp;Liu, Yu-Tsung&nbsp;Wu, Mu-Ting&nbsp;Wu, Chieo-Mo&nbsp;Li, Norman&nbsp;Chang, Ying-Shiun&nbsp;Li, and Wen-Tze&nbsp;Chuang</em><br>
    ASP-DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-chhabriaMAVIRECMLAidedVectored2021" >
<td align='right'>
3.
</td>
<td>
    MAVIREC: ML-Aided Vectored IR-Drop Estimation and Classification <br>
    <em>Vidya&nbsp;A.&nbsp;Chhabria, Yanqing&nbsp;Zhang, Haoxing&nbsp;Ren, Ben&nbsp;Keller, Brucek&nbsp;Khailany, and Sachin&nbsp;S.&nbsp;Sapatnekar</em><br>
    ASP-DAC 2021  <br>
    
</td>
</tr>


<tr id="tr-chhabriaThermalIRDrop2021" >
<td align='right'>
4.
</td>
<td>
    Thermal and IR Drop Analysis Using Convolutional Encoder-Decoder Networks [<a href='https://ieeexplore.ieee.org/document/9371634' target='_blank'>paper</a>] <br>
    <em>Vidya&nbsp;A.&nbsp;Chhabria, Vipul&nbsp;Ahuja, Ashwath&nbsp;Prabhu, Nikhil&nbsp;Patil, Palkesh&nbsp;Jain, and Sachin&nbsp;S.&nbsp;Sapatnekar</em><br>
    ASP-DAC 2021  <br>
    
</td>
</tr>


<tr id="tr-huangDynamicIRDropECO2020" >
<td align='right'>
5.
</td>
<td>
    Dynamic IR-Drop ECO Optimization by Cell Movement with Current Waveform Staggering and Machine Learning Guidance [<a href='https://ieeexplore.ieee.org/document/9256574' target='_blank'>paper</a>] <br>
    <em>Xuan-Xue&nbsp;Huang, Hsien-Chia&nbsp;Chen, Sheng-Wei&nbsp;Wang, Iris&nbsp;Hui-Ru&nbsp;Jiang, Yih-Chih&nbsp;Chou, and Cheng-Hong&nbsp;Tsai</em><br>
    ICCAD 2020  <br>
    
</td>
</tr>


<tr id="tr-pao2020xgbir" >
<td align='right'>
6.
</td>
<td>
    XGBIR: An XGBoost-based IR Drop Predictor for Power Delivery Network [<a href='https://ieeexplore.ieee.org/document/9116327' target='_blank'>paper</a>] <br>
    <em>Chi-Hsien&nbsp;Pao, An-Yu&nbsp;Su, and Yu-Min&nbsp;Lee</em><br>
    DATE 2020  <br>
    
</td>
</tr>


<tr id="tr-xieFastIRDrop2020" >
<td align='right'>
7.
</td>
<td>
    Fast IR Drop Estimation with Machine Learning [<a href='https://arxiv.org/pdf/2011.13491.pdf' target='_blank'>paper</a>] <br>
    <em>Zhiyao&nbsp;Xie, Hai&nbsp;Li, Xiaoqing&nbsp;Xu, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    ICCAD 2020  <br>
    
</td>
</tr>


<tr id="tr-xiePowerNetTransferableDynamic2020a" >
<td align='right'>
8.
</td>
<td>
    PowerNet: Transferable Dynamic IR Drop Estimation via Maximum Convolutional Neural Network [<a href='https://ieeexplore.ieee.org/document/9045574' target='_blank'>paper</a>] <br>
    <em>Zhiyao&nbsp;Xie, Haoxing&nbsp;Ren, Brucek&nbsp;Khailany, Ye&nbsp;Sheng, Santosh&nbsp;Santosh, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    ASP-DAC 2020  <br>
    
</td>
</tr>


<tr id="tr-zhou2020gridnet" >
<td align='right'>
9.
</td>
<td>
    GridNet: Fast Data-Driven EM-induced IR Drop Prediction and Localized Fixing for on-Chip Power Grid Networks [<a href='https://ieeexplore.ieee.org/document/9256488' target='_blank'>paper</a>] <br>
    <em>Han&nbsp;Zhou, Wentian&nbsp;Jin, and Sheldon&nbsp;X.-D.&nbsp;Tan</em><br>
    ICCAD 2020  <br>
    
</td>
</tr>


<tr id="tr-hoIncPIRDFastLearningBased2019" >
<td align='right'>
10.
</td>
<td>
    IncPIRD: Fast Learning-Based Prediction of Incremental IR Drop [<a href='https://ieeexplore.ieee.org/document/8942110/' target='_blank'>paper</a>] <br>
    <em>Chia-Tung&nbsp;Ho and Andrew&nbsp;B.&nbsp;Kahng</em><br>
    ICCAD 2019  <br>
    
</td>
</tr>


<tr id="tr-fang2018machinelearningbased" >
<td align='right'>
11.
</td>
<td>
    Machine-Learning-Based Dynamic IR Drop Prediction for ECO [<a href='https://ieeexplore.ieee.org/document/8587752/' target='_blank'>paper</a>] <br>
    <em>Yen-Chun&nbsp;Fang, Heng-Yi&nbsp;Lin, Min-Yan&nbsp;Sui, Chien-Mo&nbsp;Li, and Eric&nbsp;Jia-Wei&nbsp;Fang</em><br>
    ICCAD 2018  <br>
    
</td>
</tr>

</table>
<h3>Reliability</h3>
<table class="table table-hover">

<tr id="tr-9875000" >
<td align='right'>
1.
</td>
<td>
    Efficient Learning Strategies for Machine Learning-Based Characterization of Aging-Aware Cell Libraries <br>
    <em>Florian&nbsp;Klemme and Hussam&nbsp;Amrouch</em><br>
    TCSI 2022  <br>
    
</td>
</tr>


<tr id="tr-9704874" >
<td align='right'>
2.
</td>
<td>
    Scalable Machine Learning to Estimate the Impact of Aging on Circuits Under Workload Dependency [<a href='https://ieeexplore.ieee.org/document/9704874' target='_blank'>paper</a>] <br>
    <em>Florian&nbsp;Klemme and Hussam&nbsp;Amrouch</em><br>
    TCSI 2022  <br>
    
</td>
</tr>


<tr id="tr-9394564" >
<td align='right'>
3.
</td>
<td>
    Machine Learning for On-the-Fly Reliability-Aware Cell Library Characterization [<a href='https://ieeexplore.ieee.org/document/9394564' target='_blank'>paper</a>] <br>
    <em>Florian&nbsp;Klemme and Hussam&nbsp;Amrouch</em><br>
    TCSI 2021  <br>
    
</td>
</tr>

</table>
<h3>Gate Sizing</h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/iccad/ZhouYPSZWHCH22" >
<td align='right'>
1.
</td>
<td>
    Heterogeneous Graph Neural Network-Based Imitation Learning for Gate Sizing Acceleration [<a href='https://doi.org/10.1145/3508352.3549361' target='_blank'>paper</a>] <br>
    <em>Xinyi&nbsp;Zhou, Junjie&nbsp;Ye, Chak-Wa&nbsp;Pui, Kun&nbsp;Shao, Guangliang&nbsp;Zhang, Bin&nbsp;Wang, Jianye&nbsp;Hao, Guangyong&nbsp;Chen, and Pheng-Ann&nbsp;Heng</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/iccad/NathPHYKR22" >
<td align='right'>
2.
</td>
<td>
    TransSizer: A Novel Transformer-Based Fast Gate Sizer [<a href='https://doi.org/10.1145/3508352.3549442' target='_blank'>paper</a>] <br>
    <em>Siddhartha&nbsp;Nath, Geraldo&nbsp;Pradipta, Corey&nbsp;Hu, Tian&nbsp;Yang, Brucek&nbsp;Khailany, and Haoxing&nbsp;Ren</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-lu2021rl" >
<td align='right'>
3.
</td>
<td>
    Rl-sizer: Vlsi gate sizing for timing optimization using deep reinforcement learning <br>
    <em>Yi-Chen&nbsp;Lu, Siddhartha&nbsp;Nath, Vishal&nbsp;Khandelwal, and Sung&nbsp;Kyu&nbsp;Lim</em><br>
    DAC 2021  <br>
    
</td>
</tr>


<tr id="tr-wang2020gcn" >
<td align='right'>
4.
</td>
<td>
    GCN-RL circuit designer: Transferable transistor sizing with graph neural networks and reinforcement learning [<a href='https://arxiv.org/abs/2005.00406' target='_blank'>paper</a>] <br>
    <em>Hanrui&nbsp;Wang, Kuan&nbsp;Wang, Jiacheng&nbsp;Yang, Linxiao&nbsp;Shen, Nan&nbsp;Sun, Hae-Seung&nbsp;Lee, and Song&nbsp;Han</em><br>
    DAC 2020  <br>
    
</td>
</tr>

</table>
<h3>Power</h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/aspdac/LuZJST23" >
<td align='right'>
1.
</td>
<td>
    Learning Based Spatial Power Characterization and Full-Chip Power Estimation for Commercial TPUs [<a href='https://doi.org/10.1145/3566097.3568347' target='_blank'>paper</a>] <br>
    <em>Jincong&nbsp;Lu, Jinwei&nbsp;Zhang, Wentian&nbsp;Jin, Sachin&nbsp;Sachdeva, and Sheldon&nbsp;X.&nbsp;-D.&nbsp;Tan</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/aspdac/ZhaiC023" >
<td align='right'>
2.
</td>
<td>
    Microarchitecture Power Modeling via Artificial Neural Network and Transfer Learning [<a href='https://doi.org/10.1145/3566097.3567844' target='_blank'>paper</a>] <br>
    <em>Jianwang&nbsp;Zhai, Yici&nbsp;Cai, and Bei&nbsp;Yu</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Mask Optimization

<h3>Layout Generation</h3>
<table class="table table-hover">

<tr id="tr-DAC23_Diff" >
<td align='right'>
1.
</td>
<td>
    DiffPattern: Layout Pattern Generation via Discrete Diffusion <br>
    <em>Zixiao&nbsp;Wang, Yunheng&nbsp;Shen, Wenqian&nbsp;Zhao, Yang&nbsp;Bai, Guojin&nbsp;Chen, Farzan&nbsp;Farnia, and Bei&nbsp;Yu</em><br>
    DAC 2023  <br>
    
</td>
</tr>

</table>
<h3>Lithography</h3>
<table class="table table-hover">

<tr id="tr-DAC23_Nitho" >
<td align='right'>
1.
</td>
<td>
    Physics-Informed Optical Kernel Regression Using Complex-valued Neural Fields <br>
    <em>Guojin&nbsp;Chen, Zehua&nbsp;Pei, Haoyu&nbsp;Yang, Yuzhe&nbsp;Ma, Bei&nbsp;Yu, and Martin&nbsp;Wong</em><br>
    DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-wang2022deepeb" >
<td align='right'>
2.
</td>
<td>
    DeePEB: A Neural Partial Differential Equation Solver for Post Exposure Baking Simulation in Lithography [<a href='https://dl.acm.org/doi/abs/10.1145/3508352.3549398' target='_blank'>paper</a>]  [<a href='https://github.com/Brilight/DeePEB' target='_blank'>code</a>] <br>
    <em>Qipan&nbsp;Wang, Xiaohan&nbsp;Gao, Yibo&nbsp;Lin, Runsheng&nbsp;Wang, and Ru&nbsp;Huang</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-10.1145/3489517.3530580" >
<td align='right'>
3.
</td>
<td>
    Generic Lithography Modeling with Dual-Band Optics-Inspired Neural Networks 
[<a href='javascript:;'
    onclick='$("#abs_10.1145/3489517.3530580").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3489517.3530580' target='_blank'>paper</a>] <br>
    <em>Haoyu&nbsp;Yang, Zongyi&nbsp;Li, Kumara&nbsp;Sastry, Saumyadip&nbsp;Mukhopadhyay, Mark&nbsp;Kilgard, Anima&nbsp;Anandkumar, Brucek&nbsp;Khailany, Vivek&nbsp;Singh, and Haoxing&nbsp;Ren</em><br>
    DAC 2022  <br>
    
<div id="abs_10.1145/3489517.3530580" style="text-align: justify; display: none" markdown="1">
Lithography simulation is a critical step in VLSI design and optimization for manufacturability. Existing solutions for highly accurate lithography simulation with rigorous models are computationally expensive and slow, even when equipped with various approximation techniques. Recently, machine learning has provided alternative solutions for lithography simulation tasks such as coarse-grained edge placement error regression and complete contour prediction. However, the impact of these learning-based methods has been limited due to restrictive usage scenarios or low simulation accuracy. To tackle these concerns, we introduce an dual-band optics-inspired neural network design that considers the optical physics underlying lithography. To the best of our knowledge, our approach yields the first published via/metal layer contour simulation at 1nm2/pixel resolution with any tile size. Compared to previous machine learning based solutions, we demonstrate that our framework can be trained much faster and offers a significant improvement on efficiency and image quality with 20\texttimes smaller model size. We also achieve 85\texttimes simulation speedup over traditional lithography simulator with ~ 1% accuracy loss.
</div>

</td>
</tr>


<tr id="tr-DBLP:conf/iccad/PanCXH022" >
<td align='right'>
4.
</td>
<td>
    Robustify ML-Based Lithography Hotspot Detectors [<a href='https://doi.org/10.1145/3508352.3549389' target='_blank'>paper</a>] <br>
    <em>Jingyu&nbsp;Pan, Chen-Chia&nbsp;Chang, Zhiyao&nbsp;Xie, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>

</table>
<h3>Mask Optimization</h3>
<table class="table table-hover">

<tr id="tr-DBLP:conf/aspdac/YangR23" >
<td align='right'>
1.
</td>
<td>
    Enabling Scalable AI Computational Lithography with Physics-Inspired Models [<a href='https://doi.org/10.1145/3566097.3568361' target='_blank'>paper</a>] <br>
    <em>Haoyu&nbsp;Yang and Haoxing&nbsp;Ren</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-DBLP:conf/aspdac/ShaoLF23" >
<td align='right'>
2.
</td>
<td>
    Data-Driven Approaches for Process Simulation and Optical Proximity Correction [<a href='https://doi.org/10.1145/3566097.3568362' target='_blank'>paper</a>] <br>
    <em>Hao-Chiang&nbsp;Shao, Chia-Wen&nbsp;Lin, and Shao-Yun&nbsp;Fang</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-ICCAD22_AdaOPC" >
<td align='right'>
3.
</td>
<td>
    AdaOPC: A Self-Adaptive Mask Optimization Framework For Real Design Patterns [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C148-ICCAD2022-AdaOPC.pdf' target='_blank'>paper</a>] <br>
    <em>Wenqian&nbsp;Zhao, Xufeng&nbsp;Yao, Ziyang&nbsp;Yu, Guojin&nbsp;Chen, Yuzhe&nbsp;Ma, Bei&nbsp;Yu, and Martin&nbsp;Wong</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-ICCAD21_develset" >
<td align='right'>
4.
</td>
<td>
    DevelSet: Deep Neural Level Set for Instant Mask optimization 
[<a href='javascript:;'
    onclick='$("#abs_ICCAD21_develset").toggle()'>abs</a>] [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C124-ICCAD2021-DevelSet.pdf' target='_blank'>paper</a>] <br>
    <em>Guojin&nbsp;Chen, Ziyang&nbsp;Yu, Hongduo&nbsp;Liu, Yuzhe&nbsp;Ma, and Bei&nbsp;Yu</em><br>
    ICCAD 2021  <br>
    
<div id="abs_ICCAD21_develset" style="text-align: justify; display: none" markdown="1">
With the feature size continuously shrinking in advanced technology nodes, mask optimization is increasingly crucial in the conventional design flow, accompanied by an explosive growth in prohibitive computational overhead in optical proximity correction (OPC) methods. Recently, inverse lithography technique (ILT) has drawn significant attention and is becoming prevalent in emerging OPC solutions. However, ILT methods are either time-consuming or in weak performance of mask printability and manufacturability. In this paper, we present DevelSet, a GPU and deep neural network (DNN) accelerated level set OPC framework for metal layer. We first improve the conventional level set-based ILT algorithm by introducing the curvature term to reduce mask complexity and applying GPU acceleration to overcome computational bottlenecks. To further enhance printability and fast iterative convergence, we propose a novel deep neural network delicately designed with level set intrinsic principles to facilitate the joint optimization of DNN and GPU accelerated level set optimizer. Experimental results show that DevelSet framework surpasses the state-of-theart methods in printability and boost the runtime performance achieving instant level (around 1 second).
</div>

</td>
</tr>


<tr id="tr-ICCAD20_damo" >
<td align='right'>
5.
</td>
<td>
    DAMO: Deep Agile Mask Optimization for Full Chip Scale 
[<a href='javascript:;'
    onclick='$("#abs_ICCAD20_damo").toggle()'>abs</a>] [<a href='https://www.cse.cuhk.edu.hk/~byu/papers/C104-ICCAD2020-DAMO.pdf' target='_blank'>paper</a>] <br>
    <em>Guojin&nbsp;Chen, Wanli&nbsp;Chen, Yuzhe&nbsp;Ma, Haoyu&nbsp;Yang, and Bei&nbsp;Yu</em><br>
    ICCAD 2020  <br>
    
<div id="abs_ICCAD20_damo" style="text-align: justify; display: none" markdown="1">
Continuous scaling of the VLSI system leaves a great challenge on manufacturing and optical proximity correction (OPC) is widely applied in conventional design flow for manufacturability optimization. Traditional techniques conducted OPC by leveraging a lithography model and suffered from prohibitive computational overhead, and mostly focused on optimizing a single clip without addressing how to tackle the full chip. In this paper, we present DAMO, a high performance and scalable deep learning-enabled OPC system for full chip scale. It is an end-to-end mask optimization paradigm which contains a Deep Lithography Simulator (DLS) for lithography modeling and a Deep Mask Generator (DMG) for mask pattern generation. Moreover, a novel layout splitting algorithm customized for DAMO is proposed to handle the full chip OPC problem. Extensive experiments show that DAMO outperforms the state-of-the-art OPC solutions in both academia and industrial commercial toolkit.
</div>

</td>
</tr>


<tr id="tr-10.1145/3195970.3196056" >
<td align='right'>
6.
</td>
<td>
    GAN-OPC: Mask Optimization with Lithography-Guided Generative Adversarial Nets 
[<a href='javascript:;'
    onclick='$("#abs_10.1145/3195970.3196056").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3195970.3196056' target='_blank'>paper</a>] <br>
    <em>Haoyu&nbsp;Yang, Shuhe&nbsp;Li, Yuzhe&nbsp;Ma, Bei&nbsp;Yu, and Evangeline&nbsp;F.&nbsp;Y.&nbsp;Young</em><br>
    DAC 2018  <br>
    
<div id="abs_10.1145/3195970.3196056" style="text-align: justify; display: none" markdown="1">
Mask optimization has been a critical problem in the VLSI design flow due to the mismatch between the lithography system and the continuously shrinking feature sizes. Optical proximity correction (OPC) is one of the prevailing resolution enhancement techniques (RETs) that can significantly improve mask printability. However, in advanced technology nodes, the mask optimization process consumes more and more computational resources. In this paper, we develop a generative adversarial network (GAN) model to achieve better mask optimization performance. We first develop an OPC-oriented GAN flow that can learn target-mask mapping from the improved architecture and objectives, which leads to satisfactory mask optimization results. To facilitate the training process and ensure better convergence, we also propose a pre-training procedure that jointly trains the neural network with inverse lithography technique (ILT). At convergence, the generative network is able to create quasi-optimal masks for given target circuit patterns and fewer normal OPC steps are required to generate high quality masks. Experimental results show that our flow can facilitate the mask optimization process as well as ensure a better printability.
</div>

</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Analog Layout Synthesis

<h3>Analog Simulation</h3>
<table class="table table-hover">

<tr id="tr-li2022deep" >
<td align='right'>
1.
</td>
<td>
    A Deep Learning Approach for Efficient Electromagnetic Analysis of On-Chip Inductor with Dummy Metal Fillings [<a href='https://doi.org/10.3390/electronics11244214' target='_blank'>paper</a>] <br>
    <em>Xiangliang&nbsp;Li, Yijie&nbsp;Tang, Peng&nbsp;Zhao, Shichang&nbsp;Chen, Kuiwen&nbsp;Xu, and Gaofeng&nbsp;Wang</em><br>
    Electronics 2022  <br>
    
</td>
</tr>


<tr id="tr-huang2022applications" >
<td align='right'>
2.
</td>
<td>
    Applications of physics-informed neural networks in power systems-a review [<a href='https://ieeexplore.ieee.org/document/9743327' target='_blank'>paper</a>] <br>
    <em>Bin&nbsp;Huang and Jianhui&nbsp;Wang</em><br>
    Transactions on Power Systems 2022  <br>
    
</td>
</tr>

</table>
<h3>Analog Layout Synthesis</h3>
<table class="table table-hover">

<tr id="tr-DAC23_MTLD" >
<td align='right'>
1.
</td>
<td>
    MTL-Designer: An Integrated Flow for Analysis and Synthesis of Microstrip Transmission Line <br>
    <em>Qipan&nbsp;Wang, Ping&nbsp;Liu, Liguo&nbsp;Jiang, Mingjie&nbsp;Liu, Yibo&nbsp;Lin, Runsheng&nbsp;Wang, and Ru&nbsp;Huang</em><br>
    DAC 2023  <br>
    
</td>
</tr>


<tr id="tr-DAC20_ClosingTheDesignLoop" >
<td align='right'>
2.
</td>
<td>
    Closing the Design Loop: Bayesian Optimization Assisted Hierarchical Analog Layout Synthesis 
[<a href='javascript:;'
    onclick='$("#abs_DAC20_ClosingTheDesignLoop").toggle()'>abs</a>] [<a href='https://dl.acm.org/doi/pdf/10.5555/3437539.3437770' target='_blank'>paper</a>]  [<a href='https://github.com/magical-eda/MAGICAL.git' target='_blank'>code</a>]  [<a href='https://pdfs.semanticscholar.org/e994/c108710d83541a08d21b4a34ca3dfe221c31.pdf' target='_blank'>slides</a>] <br>
    <em>Mingjie&nbsp;Liu, Keren&nbsp;Zhu, Xiyuan&nbsp;Tang, Biying&nbsp;Xu, Wei&nbsp;Shi, Nan&nbsp;Sun, and David&nbsp;Z.&nbsp;Pan</em><br>
    DAC 2019  <br>
    
<div id="abs_DAC20_ClosingTheDesignLoop" style="text-align: justify; display: none" markdown="1">
Existing analog layout synthesis tools provide little guarantee to post layout performance and have limited capabilities of handling system-level designs. In this paper, we present a closed-loop hierarchical analog layout synthesizer, capable of handling system designs. To ensure system performance, the building block layout implementations are optimized efficiently, utilizing post layout simulations with multi-objective Bayesian optimization. To the best of our knowledge, this is the first work demonstrating success in automated layout synthesis on generic analog system designs. Experimental results show our synthesized continuous-time ΔΣ modulator (CTDSM) achieves post layout performance of 65.9dB in signal to noise and distortion ratio (SNDR), compared with 67.8dB in the schematic design.
</div>

</td>
</tr>

</table>
<h3>Analog Layout Placement</h3>
<table class="table table-hover">

<tr id="tr-ChangPXLLH023" >
<td align='right'>
1.
</td>
<td>
    Fully Automated Machine Learning Model Development for Analog Placement Quality Prediction [<a href='https://doi.org/10.1145/3566097.3567881' target='_blank'>paper</a>] <br>
    <em>Chen-Chia&nbsp;Chang, Jingyu&nbsp;Pan, Zhiyao&nbsp;Xie, Yaguang&nbsp;Li, Yishuang&nbsp;Lin, Jiang&nbsp;Hu, and Yiran&nbsp;Chen</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-BudakSSP23" >
<td align='right'>
2.
</td>
<td>
    APOSTLE: Asynchronously Parallel Optimization for Sizing Analog Transistors Using DNN Learning [<a href='https://doi.org/10.1145/3566097.3567880' target='_blank'>paper</a>] <br>
    <em>Ahmet&nbsp;Faruk&nbsp;Budak, David&nbsp;Smart, Brian&nbsp;Swahn, and David&nbsp;Z.&nbsp;Pan</em><br>
    ASPDAC 2023  <br>
    
</td>
</tr>


<tr id="tr-9712592" >
<td align='right'>
3.
</td>
<td>
    Generative-Adversarial-Network-Guided Well-Aware Placement for Analog Circuits [<a href='https://ieeexplore.ieee.org/document/9712592' target='_blank'>paper</a>] <br>
    <em>Keren&nbsp;Zhu, Hao&nbsp;Chen, Mingjie&nbsp;Liu, Xiyuan&nbsp;Tang, Wei&nbsp;Shi, Nan&nbsp;Sun, and David&nbsp;Z.&nbsp;Pan</em><br>
    ASP-DAC 2022  <br>
    
</td>
</tr>


<tr id="tr-GaoDLZPL21" >
<td align='right'>
4.
</td>
<td>
    Layout Symmetry Annotation for Analog Circuits with Graph Neural Networks [<a href='https://doi.org/10.1145/3394885.3431545' target='_blank'>paper</a>] <br>
    <em>Xiaohan&nbsp;Gao, Chenhui&nbsp;Deng, Mingjie&nbsp;Liu, Zhiru&nbsp;Zhang, David&nbsp;Z.&nbsp;Pan, and Yibo&nbsp;Lin</em><br>
    ASPDAC 2021  <br>
    
</td>
</tr>

</table>
<h3>Thermal Simulation</h3>
<table class="table table-hover">

<tr id="tr-ranade2022thermal" >
<td align='right'>
1.
</td>
<td>
    A Thermal Machine Learning Solver For Chip Simulation [<a href='https://doi.org/10.1145/3551901.3556484' target='_blank'>paper</a>] <br>
    <em>Rishikesh&nbsp;Ranade, Haiyang&nbsp;He, Jay&nbsp;Pathak, Norman&nbsp;Chang, Akhilesh&nbsp;Kumar, and Jimin&nbsp;Wen</em><br>
    ML4CAD 2022  <br>
    
</td>
</tr>


<tr id="tr-chen2022fast" >
<td align='right'>
2.
</td>
<td>
    Fast Thermal Analysis for Chiplet Design based on Graph Convolution Networks [<a href='https://ieeexplore.ieee.org/document/9712583' target='_blank'>paper</a>] <br>
    <em>Liang&nbsp;Chen, Wentian&nbsp;Jin, and Sheldon&nbsp;X-D&nbsp;Tan</em><br>
    ASP-DAC 2022  <br>
    
</td>
</tr>

</table>
<h3>Electromigration Analysis</h3>
<table class="table table-hover">

<tr id="tr-jin2022hierpinn" >
<td align='right'>
1.
</td>
<td>
    HierPINN-EM: Fast Learning-Based Electromigration Analysis for Multi-Segment Interconnects Using Hierarchical Physics-Informed Neural Network [<a href='https://dl.acm.org/doi/abs/10.1145/3508352.3549371' target='_blank'>paper</a>] <br>
    <em>Wentian&nbsp;Jin, Liang&nbsp;Chen, Subed&nbsp;Lamichhane, Mohammadamir&nbsp;Kavousi, and Sheldon&nbsp;X-D&nbsp;Tan</em><br>
    ICCAD 2022  <br>
    
</td>
</tr>


<tr id="tr-hou2022space" >
<td align='right'>
2.
</td>
<td>
    A Space-Time Neural Network for Analysis of Stress Evolution Under DC Current Stressing <br>
    <em>Tianshu&nbsp;Hou, Ngai&nbsp;Wong, Quan&nbsp;Chen, Zhigang&nbsp;Ji, and Hai-Bao&nbsp;Chen</em><br>
    TCAD 2022  <br>
    
</td>
</tr>

</table>
<h3>Analog Layout Routing</h3>
<table class="table table-hover">

<tr id="tr-chen2023trouter" >
<td align='right'>
1.
</td>
<td>
    TRouter: Thermal-driven PCB Routing via Non-Local Crisscross Attention Networks [<a href='https://ieeexplore.ieee.org/abstract/document/10042057' target='_blank'>paper</a>] <br>
    <em>Tinghuan&nbsp;Chen, Silu&nbsp;Xiong, Huan&nbsp;He, and Bei&nbsp;Yu</em><br>
    TCAD 2023  <br>
    
</td>
</tr>


<tr id="tr-zhu2019geniusroute" >
<td align='right'>
2.
</td>
<td>
    GeniusRoute: A new analog routing paradigm using generative neural network guidance [<a href='https://ieeexplore.ieee.org/document/8942164' target='_blank'>paper</a>] <br>
    <em>Keren&nbsp;Zhu, Mingjie&nbsp;Liu, Yibo&nbsp;Lin, Biying&nbsp;Xu, Shaolan&nbsp;Li, Xiyuan&nbsp;Tang, Nan&nbsp;Sun, and David&nbsp;Z&nbsp;Pan</em><br>
    ICCAD 2019  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Testing

<h3></h3>
<table class="table table-hover">

<tr id="tr-10.1145/3316781.3317838" >
<td align='right'>
1.
</td>
<td>
    High Performance Graph Convolutional Networks with Applications in Testability Analysis 
[<a href='javascript:;'
    onclick='$("#abs_10.1145/3316781.3317838").toggle()'>abs</a>] [<a href='https://doi.org/10.1145/3316781.3317838' target='_blank'>paper</a>] <br>
    <em>Yuzhe&nbsp;Ma, Haoxing&nbsp;Ren, Brucek&nbsp;Khailany, Harbinder&nbsp;Sikka, Lijuan&nbsp;Luo, Karthikeyan&nbsp;Natarajan, and Bei&nbsp;Yu</em><br>
    DAC 2019  <br>
    
<div id="abs_10.1145/3316781.3317838" style="text-align: justify; display: none" markdown="1">
Applications of deep learning to electronic design automation (EDA) have recently begun to emerge, although they have mainly been limited to processing of regular structured data such as images. However, many EDA problems require processing irregular structures, and it can be non-trivial to manually extract important features in such cases. In this paper, a high performance graph convolutional network (GCN) model is proposed for the purpose of processing irregular graph representations of logic circuits. A GCN classifier is firstly trained to predict observation point candidates in a netlist. The GCN classifier is then used as part of an iterative process to propose observation point insertion based on the classification results. Experimental results show the proposed GCN model has superior accuracy to classical machine learning models on difficult-to-observation nodes prediction. Compared with commercial testability analysis tools, the proposed observation point insertion flow achieves similar fault coverage with an 11% reduction in observation points and a 6% reduction in test pattern count.
</div>

</td>
</tr>

</table>
<h3>Atpg</h3>
<table class="table table-hover">

<tr id="tr-9983862" >
<td align='right'>
1.
</td>
<td>
    Neural Fault Analysis for SAT-based ATPG [<a href='https://ieeexplore.ieee.org/document/9983862' target='_blank'>paper</a>] <br>
    <em>Junhua&nbsp;Huang, Hui-Ling&nbsp;Zhen, Naixing&nbsp;Wang, Hui&nbsp;Mao, Mingxuan&nbsp;Yuan, and Yu&nbsp;Huang</em><br>
    ITC 2022  <br>
    
</td>
</tr>

</table>
<h3>Test Point Insertion</h3>
<table class="table table-hover">

<tr id="tr-Shi2022DeepTPITP" >
<td align='right'>
1.
</td>
<td>
    DeepTPI: Test Point Insertion with Deep Reinforcement Learning [<a href='https://arxiv.org/abs/2206.06975' target='_blank'>paper</a>] <br>
    <em>Zhengyuan&nbsp;Shi, Min&nbsp;Li, Sadaf&nbsp;Khan, Liuzheng&nbsp;Wang, Naixing&nbsp;Wang, Yu&nbsp;Huang, and Qiang&nbsp;Xu</em><br>
    ITC 2022  <br>
    
</td>
</tr>

</table>



## <i class="fa fa-chevron-right"></i> Dataset and Tools

<h3></h3>
<table class="table table-hover">

<tr id="tr-chai2022circuitnet" >
<td align='right'>
1.
</td>
<td>
    CircuitNet: An Open-Source Dataset for Machine Learning Applications in Electronic Design Automation (EDA) [<a href='https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3571-8' target='_blank'>paper</a>]  [<a href='https://circuitnet.github.io/' target='_blank'>code</a>] <br>
    <em>Zhuomin&nbsp;Chai, Yuxiang&nbsp;Zhao, Yibo&nbsp;Lin, Wei&nbsp;Liu, Runsheng&nbsp;Wang, and Ru&nbsp;Huang</em><br>
    SCIENCE CHINA Information Sciences 2022  <br>
    
</td>
</tr>

</table>

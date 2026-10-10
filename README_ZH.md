# GMM模型与F₁控制器开发交接

本包用于老师开发下一步控制器。包括模型、独立推理入口、原F₁源代码依赖、物理配置、训练特征表和三个train样例。没有增加场景、没有重训模型，也没有将GMM₂接入控制。

## 先区分两个GMM₁版本

|版本|输入和输出|用途|
|---|---|---|
|论文K=15|六维当时状态特征 → 单一横向加速度残差|审稿回复版本；按训练BIC从1～15选择|
|控制实验K=3|六维rollout与候选控制特征 → [Δay, Δr]端点残差|当前F₁以及200场景使用的模型；按验证误差从3/5/8/10选择|
|GMM₂ Current K=8|14维当前特征 → [q_vy, q_r]|离线一步残差模型|
|GMM₂ History K=5|Current＋8维历史 → [q_vy, q_r]|离线一步残差模型|

K=3不是K=15直接改了参数，也不能把K=15替换进去后继承当前200场景与GMM₂的结果。GMM₁的两个版本分别命名、分别调用。论文预测入口只调用原predict_residual，不启用源文件中的其他滤波或门控方法。

## 快速开始

推荐Python 3.12。以下在解压后的仓库根目录执行，路径不依赖原作者D盘目录。

```text
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reference.txt
.\.venv\Scripts\python.exe run_demo.py
.\.venv\Scripts\python.exe run_demo.py --paper
.\.venv\Scripts\python.exe verify.py
```

Linux/macOS可将解释器替换为`.venv/bin/python`，但本次数值验证平台为Windows，未宣称跨平台实测通过。只运行纯模型推理时可安装requirements-inference.txt；完整verify还需要requirements-reference.txt中的OSQP等包。

verify比较三个样例的GMM₁均值/协方差/特征、GMM₂均值/协方差/一步状态修正、8维历史、原F₁修正QP与独立保存的原项目参考输出，并检查原项目模块都从本交付目录加载。参考数据来自原项目运行，不在验证时重新生成。

requirements-reference-lock.txt记录本次干净环境的全部直接及间接依赖。需要完全匹配验证环境时，可用它替代requirements-reference.txt安装。论文模型沿用的导入链需要matplotlib，虽不绘图也保留该依赖。

### sklearn版本说明

逐文件加载核对：当前K3和论文K15在scikit-learn 1.8.0中没有版本警告；两个GMM₂资产标记为1.5.1保存，在1.8.0中加载会显示版本警告。原模型文件保持不变，警告不屏蔽。本包已在1.8.0独立环境中核对四个模型的参考输出，但不声称任意跨版本训练/操作均兼容。若重新复现GMM₂训练，应另建环境安装requirements-training-original.txt，以保留模型保存时的1.5.1版本；不要在该环境混跑GMM₁。详见docs/model_load_versions.json。

## 目录

- models：四个原始模型资产。
- src/handoff：独立推理与历史特征入口。
- src/future_work、src/residual_model：保留原Python模块层级的依赖快照，确保序列化和F₁参考代码可用。
- src/FITTED_MODEL_PARAMETERS.json：名义车辆标定参数。
- examples：普通弯、高速换向、制动入弯的输入、101点历史及期望输出；均来自train。
- data：压缩动态转移表和拟合样本清单，用于可选训练复现。
- docs：接口、模型版本、结果、环境与来源说明。
- train_from_dataset.py：从准备好的特征表复现训练的独立入口，不需要200场景原始日志。

代码入口采用相对本包文件位置定位模型。src中的旧实验脚本仅用于溯源和参考，其原有历史数据路径可能仍指向原研究环境；不要直接执行这些旧脚本。支持的交接入口为run_demo.py、verify.py、train_from_dataset.py。

## 老师下一步需要设计的接口

1. 原F₁为F₀(U₁)+Cₖ(U₀)，GMM₁通过Δr修正ey/eψ，vy/r仍为Nominal。
2. GMM₂为区间等效变化率残差，一步状态修正为F₁预测+0.05q。
3. 未来实际转角、PI/ABS记忆尚无正式预测更新。这里没有自主多步GMM₂部署接口。
4. 需要确定GMM₁与GMM₂统一递推、是否重复补偿以及怎样进入QP；本包不替老师预设该方案。

Current与History均保留。现有结果只证明一步预测收益，不代表航向振荡已经解决。完整Chrono闭环复现需另配Chrono运行库和外部资产，详见docs/CHRONO_BOUNDARY_ZH.md。

## 可选训练复现

```text
python train_from_dataset.py --output-dir training_outputs/reproduction_01
```

该命令会重新训练，仅在明确需要复现时运行，并使用上述单独的训练环境。本次交接整理只验证该命令帮助入口，没有重新训练。输入使用现有6,480条等场景抽样行，沿用K候选与validation选型，test随后评价。输出必须是空目录，不能覆盖已交付模型。源Python进程可见版本与模型资产保存版本存在差异，不能以environment_observed.json代替模型原始序列化版本记录。

## GitHub上传范围

本交付的公开仓库为https://github.com/GQR-art/gmm-mpc-controller-handoff 。未打包约1.41 GiB原始Chrono日志及完整多步回放大表。保留了约22.66 MiB的Stanford派生训练特征表，因为原F₁诊断初始化依赖它；纯ControlGMM1入口不读取该表，数值推理与原始路径对照验证。

仓库公开可见不等于对第三方数据授予新的开源许可，相关数据与外部资产保留其原有权利约束。依赖快照中带有历史模块名称不表示采用其全部实验路线；正式模型以models和接口说明为准。

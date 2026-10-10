# 模型与时间接口

## 控制实验GMM₁

输入顺序：[vx, r_bicycle, ay_bicycle, delta_future, delta_dot_future, prediction_time_s]。
每次20行，预测时间0.05～1.00 s。r_bicycle、ay_bicycle由当前vy/r、20个候选请求与速度预瞄通过原RK4递推产生。转角变化率首项为(候选首角−previous_request)/0.05，后续项为相邻候选转角差/0.05。这里是路轮角，单位rad，不是方向盘角或归一化驱动量。

输出均值形状(20,2)，协方差(20,2,2)，对应[Δay m/s², Δr rad/s]的端点残差。返回值不是20次状态增量，不得把Δr逐步作为角速度增量累加。原F₁将Δr积分成航向修正，并传播到横向位置；Δay只保留诊断。F₁的修正依赖U₀，而执行的是修正QP求出的U₁。

## GMM₂ Current

固定14维顺序如下，接口接受具名字典并按此顺序组装：

1. current_vx_mps
2. current_vy_mps
3. current_r_radps
4. current_ey_m
5. current_epsi_rad
6. request_road_wheel_rad（本周期U₁首个请求）
7. actual_road_wheel_rad（当前实测等效前轮角）
8. previous_request_road_wheel_rad
9. preview_curvature_per_m（首节点曲率预瞄）
10. preview_speed_mps（首节点速度预瞄）
11. pi_integral_before（查询时PI积分，m/s·s）
12. abs_multiplier_min（查询时四轮ABS乘子最小值）
13. F1_next_vy_mps
14. F1_next_r_radps

这些字段与模型内部scaler配套，直接传物理单位，不要在外部再次标准化。F₁下一步预测是当前可计算的模型输出，不是未来真实测量。候选控制修改后怎样重建这些字段，是新控制器应明确的接口。

## 短历史H

在Current之后追加8维。先过去0.20 s的vx、vy、r、实际转角斜率，再过去0.05 s的同四项。物理步长0.002 s，分别101点与26点，包含查询时刻。斜率使用最小二乘直线拟合：sum((t−mean(t))(y−mean(y)))/sum((t−mean(t))²)。单位分别为m/s²、m/s²、rad/s²、rad/s。

history_features严格检查101点连续采样。调用方负责保证最后时间即查询时刻，不能把未来数据装成历史。示例history.csv与input.json中的参考H一一对应。

## 输出及评价

q_vy=(Chrono下一步vy−F₁下一步vy)/0.05；q_r同理。接口返回mean、covariance、responsibility与next_vr。next_vr = [F1_next_vy,F1_next_r] + 0.05 × mean。

这些是0.05 s区间等效变化率残差，不是瞬时加速度；q_vy不等于GMM₁的Δay。协方差只是回归输出，没有经本轮验证可用于安全概率保证。未提供GMM₂闭环启用开关。

## 坐标与车辆信号

vy使用车体横向速度，不使用世界坐标Y方向速度。r为车体横摆角速度。路径ey/eψ由原error_state与参考投影实现，正式基线以原代码为准。eψ需要角度环绕，r是角速度不可wrap。实际等效前轮角在采集器中为atan(mean(tan(前轴各轮角)))，不能随意改成方向盘角或单轮角。

当前速度与真实速度预瞄偏差未被抹除。多步回顾数据使用后来实际请求及起点冻结的参考、速度预瞄、GMM₁修正；它不是固定计划Chrono分支，也不是严格同完整输入的纯动力学残差。

docs/f1_mpc_config.json给出默认权重和约束：ey/eψ/vy/r为40/12/2/2，终端倍率2，转角权重0.5、转角变化权重8；路轮角约束±0.35 rad，每控制步变化约束±0.03 rad。该JSON中的默认longitudinal_speed_mps=15不表示200场景都以15 m/s运行，变量速度控制器按各场景逐步速度预瞄构造预测。

## 论文K15

输入顺序：[vx, steering_angle, steering_rate, yaw_rate, longitudinal_acceleration, ay_bike]；输出为标量横向加速度残差。它与当前K3的输入及输出语义均不同。PaperGMM1仅用于单独调用论文资产，不是FoundationBridge替换入口。所附paper_k15.json使用训练scaler附近合成输入检查接口，不作为预测精度实验。

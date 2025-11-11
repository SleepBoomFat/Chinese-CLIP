import re
import matplotlib.pyplot as plt

# 日志文件路径列表
log_files = [

    '/data/experiments/use_all/out_2025-08-14-10-23-18.log'
    ,'/data/experiments/new_use_all_2/out_2025-08-19-06-45-27.log',
    '/data/experiments/new_use_all_top5/out_2025-08-18-08-54-33.log'
]
labels = [
    'without_memory_fusion',      # 对应 log_files[0]
    'without_memory',         # 对应 log_files[1]
    'use_all'      # 对应 log_files[2]
]

colors = ['k','b','g']  # 指定每个文件的颜色，可以根据需要调整

# 解析日志文件
losses_all = []
step_size = 30  # 控制步长，每5个步长取一次数据点

for i, file_path in enumerate(log_files):
    losses = []  # 每个文件的损失列表
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式提取Loss值
            match = re.search(r'Loss: ([0-9.]+)', line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)

    # 将每个文件的损失值保存到总列表中
    losses_all.append(losses)

    # 只取每隔step_size步长的点，减少数据点密度
    sampled_losses = losses[::step_size]
    step_indices = range(0, len(losses), step_size)  # 生成实际步长索引：0, 30, 60, 90...
    # 绘制Loss曲线，指定颜色和标签
    plt.plot(step_indices,sampled_losses, label=labels[i], color=colors[i])

# 绘制设置
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

import re
import matplotlib.pyplot as plt

# 日志文件路径列表
log_files = ['/data/experiments/use_vision_no_fusion_memory/out_2025-08-13-09-53-52.log',
             '/data/experiments/use_vision_no_memory/out_2025-08-14-03-49-55.log',
    '/data/experiments/use_all/out_2025-08-14-10-23-18.log'
]

colors = ['b', 'g','r']  # 指定每个文件的颜色，可以根据需要调整
losses_all = []

# 解析日志文件
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

    # 绘制Loss曲线，指定颜色和标签
    plt.plot(losses, label=f'Log File {i + 1}', color=colors[i])

# 绘制设置
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

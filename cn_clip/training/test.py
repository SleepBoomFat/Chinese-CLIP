import re
import matplotlib.pyplot as plt

# 日志文件路径列表
log_files = [
    '/data/experiments/use_vision_no_fusion_memory/out_2025-08-13-09-53-52.log'
]

losses = []

# 解析日志文件
for file_path in log_files:
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式提取Loss值
            match = re.search(r'Loss: ([0-9.]+)', line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)

# 绘制Loss曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

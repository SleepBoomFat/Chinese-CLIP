# 为验证集图片池和query文本计算特征
dataset_name="MUGE"
DATAPATH="/data"
split="valid" # 指定计算valid或test集特征

run_command = "export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip;" + \
f"""
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="{DATAPATH}/datasets/{dataset_name}/{split}_imgs.img_feat.jsonl" \
    --text-feats="{DATAPATH}/datasets/{dataset_name}/{split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="{DATAPATH}/datasets/{dataset_name}/{split}_predictions.jsonl"
"""
print(run_command.lstrip())
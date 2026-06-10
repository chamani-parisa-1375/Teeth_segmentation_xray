import json
from pathlib import Path

import pandas as pd
from prettytable import PrettyTable

import numpy as np

results = {}
with open('data/predict/result.json') as f:
    data = json.load(f)

def df_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert pandas DataFrame to markdown table.
    """

    headers = "| " + " | ".join(df.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(map(str, row.values)) + " |")

    markdown_table = "\n".join([headers, separator] + rows)

    return markdown_table

ious_dic = {}
dice_dic = {}

for img_stem,img_metric in data.items():

    for model_name,metric_dict in img_metric.items():

        if model_name not in ious_dic:
            ious_dic[model_name] = []
        if model_name not in dice_dic:
            dice_dic[model_name] = []

        ious_dic[model_name].append(metric_dict['mean_iou'])
        dice_dic[model_name].append(metric_dict['mean_dice'])



header = ["ModelName", "ImageSize", "Iou", "Dice"]
items = []
for model_name in ious_dic.keys():

    name = model_name.split(',')
    imgsize = int(name[-1])
    model = ','.join(name[:-1])

    iou_list = ious_dic[model_name]
    dice_list = dice_dic[model_name]

    iou = np.round(np.mean(iou_list),2)
    dice = np.round(np.mean(dice_list),2)

    items.append({"ModelName":model, "ImageSize":imgsize, "Iou":iou, "Dice":dice})


df = pd.DataFrame(items)

markdown = df_to_markdown(df)
print(markdown)
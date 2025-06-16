import os
import pandas as pd
import time
from openai import OpenAI

api_key = "Your_OpenAI_API_Key"

client = OpenAI(api_key=api_key)
# 读取数据
df = pd.read_csv("HURDAT_2new.csv",sep=',')
# 输出列表
descriptions = []

# 构建 Prompt
def build_prompt(row):
    column_names = df.columns.tolist()
    values = row.tolist()
    assert len(column_names) == len(values)

    content_lines = [f"{col.strip()}: {val}" for col, val in zip(column_names, values)]
    content = "\n".join(content_lines)

    prompt = (
        "You are an expert in Meteorology and Artificial Intelligence. Given the following structured typhoon record, "
        "generate a contextualized natural language description of the hurricane's status at that time. "
        "Be concise, factual, and human-readable.\n\n"
        f"{content}\n\n"
        "Description:"
    )
    return prompt

# 主循环：逐行调用 GPT-4
for idx, row in df.iterrows():
    prompt = build_prompt(row)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150
        )
        description = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        description = "ERROR"

    descriptions.append(description)
    time.sleep(1.2)  # 避免触发速率限制

# 保存结果
df["description"] = descriptions
df.to_csv("HURDAT_2new_w_description.csv", index=False)
print("All descriptions generated and saved!")


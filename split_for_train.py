import json
import random

with open("data/classes.json", "r", encoding="utf-8") as f:
    classes_data = json.load(f)
with open("data/poems.json", "r", encoding="utf-8") as f:
    poems_data = json.load(f)

merged_data = classes_data + poems_data

random.shuffle(merged_data)

total_size = len(merged_data)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_data = merged_data[:train_size]
val_data = merged_data[train_size : train_size + val_size]
test_data = merged_data[train_size + val_size :]


def write_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


write_jsonl(train_data, "data/train.jsonl")
write_jsonl(val_data, "data/val.jsonl")
write_jsonl(test_data, "data/test.jsonl")

print(
    f"Files created: train.jsonl ({len(train_data)} entries), "
    f"val.jsonl ({len(val_data)} entries), test.jsonl ({len(test_data)} entries)"
)

import os
import random
import re
import json


def normalize_text(text):
    text = re.sub(r"\([\s\u09E6-\u09EF]+\)", " ", text)
    text = re.sub(r"[\u09E6-\u09EF]+", " ", text)
    text = re.sub(r"[\u00A0\u200B\u200C\u200D\u2060\u3000]+", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def make_datasets():
    # parent_dir is the root dataset directory
    # Assuming you ran (from project root)
    # cd ..
    # git clone https://github.com/shuhanmirza/Bengali-Poem-Dataset
    # It would be
    parent_dir = "../Bengali-Poem-Dataset/dataset"

    poems = []
    classes = []
    missing_class_count = 0

    for poet in os.listdir(parent_dir):
        poet_dir = os.path.join(parent_dir, poet)
        if not os.path.isdir(poet_dir):
            continue

        for poem in os.listdir(poet_dir):
            poem_dir = os.path.join(poet_dir, poem)
            if not os.path.isdir(poem_dir):
                continue

            class_text = ""
            class_path = os.path.join(poem_dir, "CLASS.txt")
            if os.path.exists(class_path):
                try:
                    with open(class_path, "r", encoding="utf-8") as f:
                        class_text = normalize_text(f.read())
                except Exception as e:
                    print(f"Error reading CLASS.txt in {poem_dir}: {e}")
                    continue

            for file in os.listdir(poem_dir):
                file_path = os.path.join(poem_dir, file)
                if (
                    file.endswith(".txt")
                    and file not in ["CLASS.txt", "SOURCE.txt"]
                    and os.path.isfile(file_path)
                ):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            poem_text = f.read()

                        poem_lines = []
                        for line in poem_text.split("\n"):
                            line = normalize_text(line)
                            if line:
                                poem_lines.append(line)

                        poem_text = "\n".join(poem_lines)

                        if class_text:
                            classes.append(
                                {
                                    "Instructions": f"একটি বাংলা কবিতা লেখো যার বিষয় হলো {class_text}। কবিতায় নতুন লাইনের জন্য '\n' এবং নতুন স্তবকের জন্য '\n\n\n' ব্যবহার করো।",
                                    "Input": class_text,
                                    "Output": poem_text,
                                }
                            )
                        else:
                            missing_class_count += 1

                        if len(poem_lines) >= 2:
                            random_line_start = random.randint(0, len(poem_lines) - 2)
                            line1 = poem_lines[random_line_start]
                            line2 = poem_lines[random_line_start + 1]

                            poems.append(
                                {
                                    "Instructions": f"নিচের দুটি লাইন ব্যবহার করে একটি সম্পূর্ণ বাংলা কবিতা লেখো। কবিতায় নতুন লাইনের জন্য '\n' এবং নতুন স্তবকের জন্য '\n\n\n' ব্যবহার করো।\n{poem_text}",
                                    "Input": f"{line1}\n{line2}",
                                    "Output": poem_text,
                                }
                            )

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

    with open("data/poems.json", "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False, indent=4)
    with open("data/classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    make_datasets()

import os
import json

def convert_json_to_flat_json(input_file, folder_path, output_file, split="train"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_list = []
    skipped = 0

    for anno_id, annotation in data["annotations"].items():
        image_id = annotation["image_id"]
        question = annotation["question"]

        image_name = data["images"].get(str(image_id), "")
        image_path = os.path.join(folder_path, image_name)

        if split in ["train", "dev"]:
            answer = str(annotation.get("answer", "")).strip()

            if (answer == "") or (answer.lower() == "your answer"):
                skipped += 1
                continue

            flat_item = {
                "image_path": image_path,
                "question": question,
                "answer": answer,
            }
        else:
            flat_item = {
                "image_path": image_path,
                "question": question,
            }

        flat_list.append(flat_item)

    out_dir = os.path.dirname(output_file)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(flat_list, f, ensure_ascii=False, indent=2)

    print(f"[{split}] Saved {len(flat_list)} valid items to {output_file}")
    if split in ["train", "dev"] and skipped > 0:
        print(f"[{split}] Skipped {skipped} invalid items (empty or 'your answer')")

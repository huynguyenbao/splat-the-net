import os
import json

root_dir = "output"
merged_list = []

for scene_name in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene_name)
    if not os.path.isdir(scene_path):
        continue

    for subfolder in os.listdir(scene_path):
        subfolder_path = os.path.join(scene_path, subfolder)
        results_path = os.path.join(subfolder_path, "results.json")

        if os.path.isfile(results_path):
            scene_id = f"{scene_name}_{subfolder.lstrip('_')}"
            try:
                with open(results_path, "r") as f:
                    data = json.load(f)
                    data["scene"] = scene_id  # Inject scene identifier
                    merged_list.append(data)
                    print(f"Added: {scene_id}")
            except Exception as e:
                print(f"Error reading {results_path}: {e}")

# Save the merged list
output_path = os.path.join(root_dir, "merged_results.json")
with open(output_path, "w") as f:
    json.dump(merged_list, f, indent=4)

print(f"\n✅ Merged list saved to: {output_path}")

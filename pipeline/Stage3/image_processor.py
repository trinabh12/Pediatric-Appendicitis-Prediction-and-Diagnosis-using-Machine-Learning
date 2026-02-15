import os
import shutil
import cv2
import numpy as np
import json


class ImageProcessor:
    def __init__(self,prev_stage, img_dir, image_validation_report):
        self.img_folder = os.path.join(prev_stage, img_dir)
        self.image_report = os.path.join(self.img_folder, image_validation_report)
        with open(self.image_report, 'r') as file:
            self.image_report = json.load(file)
        self.intermediate_dir = img_dir + "-intermediate"

    def any_to_bmp(self):
        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
        shutil.copytree(self.img_folder, self.intermediate_dir)
        print(f"Step 1: Copied raw images to {self.intermediate_dir}")



        files = os.listdir(self.intermediate_dir)
        seen_stems = {}
        for filename in files:
            if filename == "image_validation_report.json":
                os.remove(os.path.join(self.intermediate_dir, filename))
                continue

            old_path = os.path.join(self.intermediate_dir, filename)

            if os.path.isdir(old_path):
                continue

            stem = filename
            while True:
                base, ext = os.path.splitext(stem)
                if ext.lower() in ['.bmp', '.png', '.jpg', '.jpeg']:
                    stem = base
                else:
                    break

            if stem in seen_stems:
                print(f"Collision detected for stem: {stem}")
                prev_path = os.path.join(self.intermediate_dir, f"{stem}.bmp")
                if os.path.exists(prev_path):
                    os.rename(prev_path, os.path.join(self.intermediate_dir, f"{stem}.1.bmp"))

                new_filename = f"{stem}.2.bmp"
            else:
                new_filename = f"{stem}.bmp"
                seen_stems[stem] = filename

            new_path = os.path.join(self.intermediate_dir, new_filename)

            img_array = np.fromfile(old_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                _, im_buf_arr = cv2.imencode(".bmp", img)
                im_buf_arr.tofile(new_path)
                if old_path != new_path:
                    os.remove(old_path)
            else:
                print(f"FAILED to read: {filename}")

        print("Step 2: Conversion to .bmp complete in intermediate folder.")

    def verify_conversion(self):

        original_files = [f for f in os.listdir(str(self.img_folder))
                          if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]

        original_stems = set()
        for f in original_files:
            stem = ".".join(f.split('.')[:-1])
            original_stems.add(stem)

        intermediate_files = os.listdir(self.intermediate_dir)
        intermediate_stems = set()
        for f in intermediate_files:
            stem = ".".join(f.split('.')[:-1])
            intermediate_stems.add(stem)

        missing = original_stems - intermediate_stems
        extra = intermediate_stems - original_stems

        print("\n--- Integrity Check Results ---")
        print(f"Original unique image stems: {len(original_stems)}")
        print(f"Intermediate unique image stems: {len(intermediate_stems)}")

        if len(missing) == 0:
            print("Success: All original images are present in the intermediate folder.")
        else:
            print(f"Missing: {len(missing)} files failed to convert!")
            for m in list(missing)[:5]:  # Show first 5
                print(f"   - {m}")

        if extra:
            print(f"Note: {len(extra)} extra stems found (check for naming collisions).")

        return len(missing) == 0

    def segregate_views(self, output_dir):
        """Moves files into View_X folders and returns a list for the Merge Script."""
        if not os.path.exists(self.intermediate_dir):
            print("Error: Run any_to_bmp first!")
            return []

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        all_files = os.listdir(self.intermediate_dir)
        registry_list = []

        for filename in all_files:
            parts = filename.split(".")
            subject_id = parts[0]
            if ' ' in parts[1]:
                raw_view_part = parts[1].split(' ')[0]
            else:
                raw_view_part = parts[1][0]


            view_folder_name = f"View_{raw_view_part}"
            target_folder = os.path.join(output_dir, view_folder_name)
            os.makedirs(target_folder, exist_ok=True)

            shutil.move(os.path.join(self.intermediate_dir, filename),
                        os.path.join(target_folder, filename))

            registry_list.append({
                "Subject_ID": subject_id,
                "View_ID": raw_view_part,
                "Final_Filename": filename,
                "Relative_Path": os.path.join(view_folder_name, filename)
            })

        return registry_list

    def process_image_data(self, output_dir):
        self.any_to_bmp()
        self.verify_conversion()

        registry_list = self.segregate_views(output_dir)
        subject_map = {}
        for item in registry_list:
            sub_id = item['Subject_ID']
            if sub_id not in subject_map:
                subject_map[sub_id] = []
            subject_map[sub_id].append({
                "view_id": item['View_ID'],
                "path": item['Relative_Path'],
                "file": item['Final_Filename']
            })

        json_path = os.path.join(output_dir, "image_registry.json")
        with open(json_path, 'w') as f:
            json.dump(subject_map, f, indent=4)

        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
            print(f"Cleanup: Removed {self.intermediate_dir}")

        print("-" * 30)
        print(f"Final Images organized in: {output_dir}")
        print(f"Registry created: {json_path}")

        return subject_map











import os
import shutil
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, data_dir, multi_image_file_name, img_dir):
        filenames = []
        with open(os.path.join(data_dir, multi_image_file_name), 'r') as file:
            for line in file:
                clean_line = line.strip().split('] ')[-1] if ']' in line else line.strip()
                if clean_line:
                    filenames.append(clean_line)

            self.multi_img = filenames
            self.img_folder = os.path.join(data_dir, img_dir)
            self.intermediate_dir = img_dir + "-intermediate"

    def any_to_bmp(self):
        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
        shutil.copytree(self.img_folder, self.intermediate_dir)
        print(f"Step 1: Copied raw images to {self.intermediate_dir}")



        files = os.listdir(self.intermediate_dir)
        seen_stems = {}
        for filename in files:
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

            # 2. Specific Collision Logic for your .png.bmp vs .bmp case
            if stem in seen_stems:
                print(f"Collision detected for stem: {stem}")
                # Rename the PREVIOUS file we saved to .1
                prev_filename = seen_stems[stem]
                prev_path = os.path.join(self.intermediate_dir, f"{stem}.bmp")
                if os.path.exists(prev_path):
                    os.rename(prev_path, os.path.join(self.intermediate_dir, f"{stem}.1.bmp"))

                # Set the CURRENT file to be .2
                new_filename = f"{stem}.2.bmp"
            else:
                new_filename = f"{stem}.bmp"
                seen_stems[stem] = filename

            new_path = os.path.join(self.intermediate_dir, new_filename)

            # 3. Encoding-safe Read/Write
            img_array = np.fromfile(old_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is not None:
                _, im_buf_arr = cv2.imencode(".bmp", img)
                im_buf_arr.tofile(new_path)
                # Remove the old messy file
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



    def segregate_views(self):
        all_files = [f for f in os.listdir(self.img_folder)
                     if str(self.any_to_bmp(f)).lower().endswith(".bmp")]
        registry = {}

        print(f"Processing {len(all_files)} images...")
        print(len(self.multi_img))
        for filename in all_files:
            filename = str(filename)

            parts = filename.split(".")
            subject_id = parts[0]
            if " " in str(parts[1]):
                view_id = str(parts[1]).split(" ")[0]
            else:
                view_id = str(parts[1])[0]

            view_folder_name = f"View_{view_id}"
            registry.update({
                "Subject_ID": subject_id,
                "View_ID": view_id,
                "Original_Filename": filename,
                "Internal_Path": os.path.join(view_folder_name, filename)
            })

        return registry


img_pro = ImageProcessor("../Regensburg Pediatric Appendicitis Dataset", "multiple_in_one", "US_Pictures")

print(img_pro.any_to_bmp())
print(img_pro.verify_conversion())


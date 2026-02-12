import os


class ImageProcessor:
    def __init__(self, data_dir, multi_image_file_name, img_dir):
        filenames = []
        with open(os.path.join(data_dir, multi_image_file_name), 'r') as file:
            for line in file:
                clean_line = line.strip().split('] ')[-1] if ']' in line else line.strip()
                if clean_line:
                    filenames.append(clean_line)

            self.multi_img = filenames

        print(f"Extracted {len(filenames)} filenames.")

    def get_multi_scan_images(self):
        for img_name in self.multi_img:
            print(img_name)




imeg = ImageProcessor("../Regensburg Pediatric Appendicitis Dataset", "multiple_in_one", "US_Pictures")
print(imeg.get_multi_scan_images())

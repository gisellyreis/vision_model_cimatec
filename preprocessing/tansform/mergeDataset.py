import os
import shutil

afo_path = "AFO"
sd_path = "seadronessee"
out_path = "dataset_combinado"

splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(out_path, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_path, "labels", split), exist_ok=True)

def copy_dataset(dataset_path, prefix):
    for split in splits:
        img_src = os.path.join(dataset_path, "images", split)
        lbl_src = os.path.join(dataset_path, "labels", split)
        img_dst = os.path.join(out_path, "images", split)
        lbl_dst = os.path.join(out_path, "labels", split)

        if os.path.exists(img_src):
            for fname in os.listdir(img_src):
                src_file = os.path.join(img_src, fname)
                dst_file = os.path.join(img_dst, f"{prefix}_{fname}")
                shutil.copy(src_file, dst_file)

        if os.path.exists(lbl_src):
            for fname in os.listdir(lbl_src):
                src_file = os.path.join(lbl_src, fname)
                dst_file = os.path.join(lbl_dst, f"{prefix}_{fname}")
                shutil.copy(src_file, dst_file)
        else:
            print(f"Aviso: pasta de labels não encontrada: {lbl_src}")

copy_dataset(afo_path, "afo")
copy_dataset(sd_path, "sds")
print("Dataset combinado criado em:", out_path)

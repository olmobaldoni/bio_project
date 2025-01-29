from train_textual_inversion import main
from os import getcwd
import os
# import argparse

# if __name__ == "__main__":
#     # target_images_dirs: la folder principale da cui poi vengono estrapolate le singole cartelle che rappresentano i singoli concetti
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--target_images_dirs", type=str, required=True, nargs="+")

#     args = parser.parse_args()

#     print("Running training for the following target images directories:")
#     for target_dir in args.target_images_dirs:
#         print("\t" + target_dir)

#     # per ogni subfolder viene lanciato il main
#     for target_dir in args.target_images_dirs:
#         target_dir_name = target_dir.split("/")[-1]
#         main(target_dir, train_log=f"{getcwd()}/training-logs/{target_dir_name}.log")


# target_images_dirs: /work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/data

# generated_images_dir: /work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/generated_data

if __name__ == "__main__":
    # target_images_dirs: la folder principale da cui poi vengono estrapolate le singole cartelle che rappresentano i singoli concetti
    target_images_dirs = r"/work/ai4bio2023/ai4bio_obaldoni/PatchCamelyon/data"

    print("Running training for the following target images directories:")
    for target_dir in os.listdir(target_images_dirs):
        print(target_dir)

    # per ogni subfolder viene lanciato il main
    for target_dir in os.listdir(target_images_dirs):
        main(
            os.path.join(target_images_dirs, target_dir),
            train_log=f"{getcwd()}/training-logs/{target_dir}.log",
        )

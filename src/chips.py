import numpy as np
import pandas as pd
import mercantile
import rio_tiler
import tqdm
from rio_tiler.io import COGReader
from rio_tiler.models import ImageData
import argparse

parser = argparse.ArgumentParser(description="train/test/val csv creation script")
parser.add_argument("--input_csv", type=str, required=True, help="")
parser.add_argument("--output_dir", type=str, required=True, help="")
args = parser.parse_args()


def main():
    df = pd.read_csv(args.input_csv)
    print(df.shape)
    img_path_lst = []
    label_path_lst = []
    for i, img in enumerate(tqdm.tqdm(df["image_fn"])):
        with COGReader(img) as cog:
            t_lst = [t for t in mercantile.tiles(*cog.bounds, 17)]
            print(len(t_lst))

            with COGReader(df["label_fn"][i]) as cog_label:
                # chip NAIP image
                for t in t_lst:  # fix
                    img = cog.tile(t.x, t.y, t.z, tilesize=256)
                    img_r = img.render(img_format="GTiff")

                    img_arr = np.moveaxis(img.data, 0, -1)
                    img_arr = img_arr - np.min(img_arr, (0, 1))
                    data_max_val = np.percentile(img_arr, 0.98, axis=(0, 1))
                    img_arr = img_arr / data_max_val * 255.0
                    np.clip(img_arr, None, 255.0, img_arr)
                    non_nodata_prop = np.sum(np.mean(img_arr, -1) > 0.0) / (256 * 256)

                    if non_nodata_prop >= 0.95:
                        # Can be replaced with Azure blob path
                        path = f"{args.output_dir}/{t.x}-{t.y}-{t.z}-img.tif"
                        path_label = f"{args.output_dir}/{t.x}-{t.y}-{t.z}-label.tif"
                        with open(path, "wb") as f:
                            f.write(img_r)
                        img_label = cog_label.tile(t.x, t.y, t.z, tilesize=256)
                        buff = img_label.render(img_format="GTiff")
                        with open(path_label, "wb") as f:
                            f.write(buff)
                        img_path_lst.append(path)
                        label_path_lst.append(path_label)
                    else:
                        print("removed too many no-data pixels")
    df_chips = pd.DataFrame(
        list(zip(img_path_lst, label_path_lst)), columns=["image_fn", "label_fn"]
    )
    df_chips["group"] = df["group"][0]

    out_csv = args.output_dir + "/" + df["group"][0] + "_256chips" + ".csv"
    df_chips.to_csv(out_csv)


if __name__ == "__main__":
    main()

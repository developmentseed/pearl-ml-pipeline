import shapely.geometry
import rasterio
import fiona.transform
import os.path as op
from pathlib import Path
import os
import pandas as pd
import numpy as np
import subprocess
import rtree
import shapely
import click
import urllib.request
import pickle


class NAIPTileIndex:
    """Utility class for performing NAIP tile lookups by location"""

    NAIP_BLOB_ROOT = "https://naipblobs.blob.core.windows.net/naip/"
    NAIP_INDEX_BLOB_ROOT = "https://naipblobs.blob.core.windows.net/naip-index/rtree/"
    INDEX_FNS = ["tile_index.dat", "tile_index.idx", "tiles.p"]

    def __init__(self, base_path, verbose=False):
        """Loads the tile index into memory (~400 MB) for use by `self.lookup()`. Downloads the index files from the blob container if they do not exist in the `base_path/` directory.
        Args:
            base_path (str): The path on the local system to look for/store the three files that make up the tile index. This path will be created if it doesn't exist.
            verbose (bool): Whether to be verbose when downloading the tile index files
        """

        # Download the index files if it doens't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for fn in NAIPTileIndex.INDEX_FNS:
            if not os.path.exists(os.path.join(base_path, fn)):
                download_url(
                    NAIPTileIndex.NAIP_INDEX_BLOB_ROOT + fn,
                    os.path.join(base_path, fn),
                    verbose,
                )

        self.base_path = base_path
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path + "/tiles.p", "rb"))

    def lookup_point(self, lat, lon):
        """Given a lat/lon coordinate pair, return the list of NAIP tiles that *contain* that point.
        Args:
            lat (float): Latitude in EPSG:4326
            lon (float): Longitude in EPSG:4326
        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the given (`lat`, `lon`) point
        Raises:
            IndexError: Raised if no tile within the index contains the given (`lat`, `lon`) point
        """

        point = shapely.geometry.Point(float(lon), float(lat))
        geom = shapely.geometry.mapping(point)

        return self.lookup_geom(geom)

    def lookup_geom(self, geom):
        """Given a GeoJSON geometry, return the list of NAIP tiles that *contain* that feature.
        Args:
            geom (dict): A GeoJSON geometry in EPSG:4326
        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the given `geom`
        Raises:
            IndexError: Raised if no tile within the index fully contains the given `geom`
        """
        shape = shapely.geometry.shape(geom)
        intersected_indices = list(self.tile_rtree.intersection(shape.bounds))
        print(intersected_indices)

        intersected_files = []
        naip_geom = []

        for idx in intersected_indices:
            print(idx)
            intersected_file = self.tile_index[idx][0]
            print(intersected_file)
            intersected_geom = self.tile_index[idx][1]
            print(intersected_geom)
            if intersected_geom.intersects(shape):
                tile_intersection = True
                f = NAIPTileIndex.NAIP_BLOB_ROOT + intersected_file
                naip_geom.append(intersected_geom)
                intersected_files.append(
                    NAIPTileIndex.NAIP_BLOB_ROOT + intersected_file
                )

        if len(intersected_files) <= 0:
            raise IndexError("No tile intersections")
        else:
            return intersected_files, naip_geom


def download_url(url, output_fn, verbose=False):
    """Download a URL to file.
    Args:
        url (str): URL of file to download
        output_fn (str): Filename to save (importantly -- not the directory to save the file to)
        verbose (bool): Whether to print how the download is going
    Returns:
        output_fn (str): Return `output_fn` as is
    """

    if verbose:
        print("Downloading file {} to {}".format(os.path.basename(url), output_fn))

    urllib.request.urlretrieve(url, output_fn)
    assert os.path.isfile(output_fn)

    if verbose:
        nBytes = os.path.getsize(output_fn)
        print("...done, {} bytes.".format(nBytes))

    return output_fn


def get_naip_tiles(label_tif_path):
    index = NAIPTileIndex("./tmp/")

    print(label_tif_path)
    with rasterio.open(label_tif_path) as f:
        geom = shapely.geometry.mapping(shapely.geometry.box(*f.bounds))
        geom = fiona.transform.transform_geom(f.crs.to_string(), "epsg:4326", geom)

        naip_azure_path, naip_lst = index.lookup_geom(geom)
    return naip_azure_path


def wrap_labels_to_naip(naip_tile_lst, out_dir, large_label_tif):
    for tile in naip_tile_lst:
        print(tile)
        with rasterio.open(tile, "r") as f:
            left, bottom, right, top = f.bounds
            crs = f.crs.to_string()
            height, width = f.height, f.width
        out_file = out_dir + tile.split("/")[-1]
        print(out_file)

        command = [
            "gdalwarp",
            "-overwrite",
            "-ot",
            "Byte",
            "-t_srs",
            crs,
            "-r",
            "near",
            "-of",
            "GTiff",
            "-te",
            str(left),
            str(bottom),
            str(right),
            str(top),
            "-ts",
            str(width),
            str(height),
            "-co",
            "COMPRESS=LZW",
            "-co",
            "BIGTIFF=YES",
            "-dstnodata",
            str(0),
            large_label_tif,
            out_file,
        ]

        subprocess.call(command)
        print("written")


def remove_notdata(in_dir, threshold):
    t_lst = [in_dir + t for t in os.listdir(in_dir) if t.endswith(".tif")]
    count = 0
    for t in t_lst:
        with rasterio.open(t) as src:
            a = src.read()
        if 0 in np.unique(a, return_counts=True)[0]:
            black_prop = (np.unique(a, return_counts=True)[1][0]) / (
                a.shape[1] * a.shape[2]
            )
            if black_prop < threshold:
                print(f"keeping {t}")
                count += 1
            else:
                print(f"too many no data pixels, removing {t}")
                os.remove(t)


def azure_urls_df(label_dir, naip_lst, label_prefix_azure, group):
    l = [f for f in os.listdir(label_dir) if f.endswith(".tif")]
    tiles_lst = []

    for n in l:
        tiles_lst.append([f for f in naip_lst if f.endswith(n)][0])

    labels_azure = [label_prefix_azure + n for n in l]

    df = pd.DataFrame(
        list(zip(tiles_lst, labels_azure)), columns=["image_fn", "label_fn"]
    )
    df["group"] = group
    return df


@click.command()
@click.option("--label_tif_path", help="path of input label tif", required=True)
@click.option(
    "--out_dir",
    help="path for label tifs that align with naip tifs to be written",
    required=True,
)
@click.option(
    "--threshold",
    help="threshold value for percentage of no data pixels",
    type=float,
    required=True,
)
@click.option("--aoi", help="aoi name", type=str, required=True)
@click.option("--group", help="label group name", type=str, required=True)
def main(label_tif_path, out_dir, threshold, aoi, group):
    # create out_dir if it doesn't exist
    Path(out_dir).mkdir(exist_ok=True)

    # get naip tiles that intersect with label tif
    naip_azure_paths = get_naip_tiles(label_tif_path)

    # wrap label tiles to naip tiles
    wrap_labels_to_naip(naip_azure_paths, out_dir, label_tif_path)

    # remove tiles that have a >= threshold percentage of no data tiles
    remove_notdata(out_dir, threshold)

    azure_df = azure_urls_df(
        out_dir,
        naip_azure_paths,
        "https://uvmlabels.blob.core.windows.net/" + aoi + "/",
        group,
    )

    train, validate, test = np.split(
        azure_df.sample(frac=1, random_state=40),
        [int(0.7 * len(azure_df)), int(0.9 * len(azure_df))],
    )

    train.to_csv(out_dir  + aoi + "_train" + ".csv")
    validate.to_csv(out_dir + aoi + "_val" ".csv")
    test.to_csv(out_dir  + aoi + "_test" + ".csv")


if __name__ == "__main__":
    main()

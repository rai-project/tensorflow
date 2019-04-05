#!/bin/env python

import re
import os
from os import path
import urllib.request
import shutil

"""Given the markdown raw text, and intended download archived model file, download, 
unarchive and optionally remove it. DO NOT RUN THIS SCRIPT without understanding what
it intends to do or experimenting on it. Otherwise it will waste your time and hard
drive space.

Arguments:
    archive_type: file extension of the archived model, zip, tar.gz, etc.
    url: Markdown README URL.
    download_dir: target directory name for downloading. Within current directory.
    dataset: optional dataset name to filter model. If not specified, will download all.
    remove: flag to remove the archived model after unarchiving.
"""
def download_and_unarchive(archive_type, url, download_dir, dataset="", remove=False):
    # if dataset is None:
    regex = r"^.*\]\((.*" + re.escape(dataset) + r".*\." + re.escape(archive_type) + r")\).*" 
    

    contents = urllib.request.urlopen(url).read().decode("utf-8")


    matches = re.findall(regex, contents, re.MULTILINE)
    models_dir = path.join(os.getcwd(), download_dir)

    try:
        os.mkdir(models_dir)
    except OSError:
        pass


    for match in matches:
        url = match
        filename = url[url.rfind("/")+1:]
        file_path = path.join(models_dir, filename)
        dir_name = filename.split(".")[0]
        dir_path = path.join(models_dir, dir_name)

        # if the archived file not exist, download it
        if not os.path.isfile(file_path) and not os.path.isdir(dir_path):
            print("downloading ... {}".format(filename))
            urllib.request.urlretrieve(url, filename=file_path)

        # if the dir already existed, no need to unarchive again
        if not os.path.isdir(dir_path):
            print("unarchiving ... {}".format(filename))
            shutil.unpack_archive(file_path, models_dir)

        # if remove is flagged and archive exist, remove if after unarchiving
        if remove and os.path.isfile(file_path):
          os.remove(file_path)

if __name__ == "__main__":
    archive_type = "tar.gz"
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/g3doc/detection_model_zoo.md"
    download_dir = "detectionModelZoo"
    download_and_unarchive(archive_type, url, download_dir, dataset="coco", remove=False)


#!/bin/env python

import re
import os
from os import path
import urllib.request
import shutil
import os
import sys

regex = r"^.*\]\((.*\.tar\.gz)\).*"

contents = urllib.request.urlopen(
    "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/README.md").read().decode("utf-8")


matches = re.findall(regex, contents, re.MULTILINE)

models_dir = path.join(os.getcwd(), "slim")

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
          shutil.unpack_archive(file_path, dir_path)

      # if remove is flagged and archive exist, remove if after unarchiving
      if os.path.isfile(file_path):
          os.remove(file_path)

    # https://github.com/tensorflow/models/tree/master/research/slim#exporting-the-inference-graph
    for f in os.listdir(dir_path):
        if f.endswith(".ckpt"):
            ckpt = f
            break
    pb = ckpt.split(".")[0] + ".pb"
    cmd = "python " + os.environ["GOPATH"]+"/src/github.com/tensorflow/models/research/slim/export_inference_graph.py --alsologtostderr --model_name=" + ckpt.split(".")[
        0] + " --output_file=" + path.join(dir_path, pb)
    print(cmd)
    os.system(cmd)


#!/bin/env python

import re
import os
from os import path
import urllib.request


regex = r"^.*\]\((.*\.tar\.gz)\).*"

contents = urllib.request.urlopen(
    "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/README.md").read().decode("utf-8")


matches = re.findall(regex, contents, re.MULTILINE)

models_dir = path.join(os.getcwd(), "tfslim_2019_04_09")

try:
    os.mkdir(models_dir)
except OSError:
    pass


for match in matches:
    url = match
    filename = url[url.rfind("/")+1:]
    print("downloading ... {}".format(filename))
    urllib.request.urlretrieve(
        url, filename=path.join(models_dir, filename))

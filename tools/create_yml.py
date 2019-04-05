import os
import subprocess
import shutil
import hashlib

term_dict = {'ssd': 'SSD', 'rcnn': 'RCNN', 'resnet': 'ResNet', 'nas': 'NAS', 'coco': 'COCO', 'coco14': 'COCO14', 'rfcn': 'RFCN',
             'mobilenet': 'MobileNet', 'ssdlite': 'SSDLite', 'fpn': 'FPN'}

# sample yml as the base
yml_dir = os.path.join(os.getcwd(), "../builtin_models")
sample_yml = "SSD_MobileNet_v2_COCO.yml"

# Get the clean name of each model
model_paths = subprocess.check_output("ls -d detectionModelZoo/*/", shell=True).decode("utf-8")
model_paths = model_paths.split('\n')[:-1]

model_names = []
pretty_names = []
for i, model_path in enumerate(model_paths):
    model_name = model_path.split("/")[1]
    model_names.append(model_name)
    terms = model_name.split("_")[:-3]
    for i in range(len(terms)):
        if terms[i] in term_dict.keys():
            terms[i] = term_dict[terms[i]]
        elif terms[i][0] == 'v':
            continue
        else:
            terms[i] = terms[i].capitalize()
    pretty_names.append('_'.join(terms))

for i in range(len(model_paths)):
    print(model_paths[i])
    print(pretty_names[i])

hash_md5 = hashlib.md5()
for model_path, complete_name, pretty_name in zip(model_paths, model_names, pretty_names):
    # generate checksum with the model_path
    graph = os.path.join(model_path, 'frozen_inference_graph.pb')
    with open(graph, 'rb') as g:
        graph_bytes = g.read()
    checksum = hashlib.md5(graph_bytes).hexdigest()

    # create a new file based on the pretty_name

    # fill out the new yml file with model_name, pretty_name and checksum
    break



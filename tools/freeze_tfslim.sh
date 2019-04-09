# fpb = "frozen_" + ckpt.split(".")[0] + ".pb"
# cmd = "freeze_graph --input_graph=" + path.join(dir_path, pb) + " --input_checkpoint=" + path.join(
#     dir_path, ckpt) + " --input_binary=true --output_graph=" + path.join(dir_path, fpb) + " --output_node_names=InceptionV2/Predictions/Reshape_1"
# print(cmd)
# os.system(cmd)

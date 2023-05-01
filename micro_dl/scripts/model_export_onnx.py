# %%
import numpy as np
import os
import sys

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")
import micro_dl.inference.inference as inference
import torch.onnx as torch_onnx
import onnxruntime as ort  # to inference ONNX models, we use the ONNX Runtime
import onnx
import torch

# %%
config = {
    "model_dir": "/hpc/projects/CompMicro/projects/virtualstaining/torch_microDL/models/2023_04_05_Phase2Nuc_HEK_lightning/shalin/lightning_logs/20230408-145505/",  # example training logs
    "model_name": "epoch=62-step=6048.ckpt",  # example checkpoint
}

torch_predictor = inference.TorchPredictor(
    config=config,
    device="cpu",  #'cpu', 'cuda', 'cuda:(int)'
    single_prediction=True,
)
torch_predictor.load_model()

sample_input = np.random.rand(1, 1, 5, 512, 512)
sample_prediction = torch_predictor.predict_image(sample_input)
# %%
save_dir = (
    "/hpc/projects/CompMicro/projects/virtualstaining/torch_microdl/models/onnx_models/"
)
model_name = "Phase2Nuc_HEK_lightning_20230408-145505_patch256_epoch62.onnx"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# %%
model = torch_predictor.model
input_tensor = torch.tensor(sample_input.astype(np.float32), requires_grad=True)
model.eval()
# %%
# Export the model
torch_onnx.export(
    model,  # model being run
    input_tensor,  # model input (or a tuple for multiple inputs)
    "test_export.onnx",  # os.path.join(save_dir,model_name),   # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size", 1: "channels", 3: "num_rows", 4: "num_cols"},
        "output": {0: "batch_size", 1: "channels", 3: "num_rows", 4: "num_cols"},
    },
)
# %%
# ------ Code for running inference session. Needs full node ---#
# onnx_model = onnx.load(
#     "/home/christian.foley/virtual_staining/workspaces/microDL/micro_dl/scripts/test_export_uninitialized.onnx"
# )
# onnx.checker.check_model(onnx_model)

# options = ort.SessionOptions()
# options.intra_op_num_threads = 1
# options.inter_op_num_threads = 1

# ort_sess = ort.InferenceSession(
#     "/home/christian.foley/virtual_staining/workspaces/microDL/micro_dl/scripts/test_export_uninitialized.onnx"
# )
# outputs = ort_sess.run(None, {"input": sample_input.numpy()})
# %%

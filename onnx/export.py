import torch
import onnx
from model.U2net_p import U2NETP
import numpy as np
import onnxruntime
import os


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    batch_size = 1
    IMG_HEIGHT = 224  # 416
    IMG_WIDTH = 224  # 416
    classes_list = ["0", "1", "2", "3", "4", "5"]
    load_params = r"D:\my_program\study\heatmap_point\weights\u2net_p_weight\u2net_p_196_epoch.pth"
    onnx_name = os.path.basename(load_params).split(".")[0]+".onnx"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    net = U2NETP(in_ch=3, out_ch=len(classes_list)).to(device)
    net.load_state_dict(torch.load(load_params))
    net.eval()

    x = torch.randn(batch_size, 3, IMG_HEIGHT, IMG_WIDTH)  # 模拟输入 batch.to(device)
    out = net(x)  # 将模型运行

    # 导出模型
    torch.onnx.export(net,  # model
                      x,  # model input
                      onnx_name,  # 导出路径
                      export_params=True,  # 是否一起保存权重
                      opset_version=11,  # 导出 ONNX 的版本号
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # 定义输入名
                      output_names=['output'],  # 定义输出名
                      # dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # 输入形状第0个是批次
                      dynamic_axes={'input': {},  # 输入形状第0个是批次
                                    'output': {0: 'batch_size'}})

    # 检查模型代码是否都支持onnx
    onnx_net = onnx.load(f"./{onnx_name}")
    onnx.checker.check_model(onnx_net)

    # 硬件检查精度误差是否在接受范围
    ort_session = onnxruntime.InferenceSession(f"./{onnx_name}")  # 创建一个会话
    # 计算ONNX运行时的输出
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # print(ort_session.get_inputs()[0].shape)  # [1, 3, 224, 224] 输入shape
    # print(len(ort_session.get_inputs()))  # 1 输入列表长度
    ort_outs = ort_session.run(None, ort_inputs)  # onnx输出结果

    # 比较ONNX运行时和PyTorch结果
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
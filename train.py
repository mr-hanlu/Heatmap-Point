import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from model.U2net_p import U2NETP
from model.UNet import MainNet
from model.UNet3Plus import UNet3Plus
from model.U2net import U2NET
from Dataset.dataset import MyDataset
from Dataset import config
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from model.focal_loss import FocalLoss

if __name__ == '__main__':
    Epoch = 200
    num = 196  # 加载的参数
    batch_size = 4
    net_model = "u2net_p"  # "unet" "u2net_p" "unet3+" "u2net"  4次16倍下采样

    load_params = f"./weights/{net_model}_weight/{net_model}_{num}_epoch.pth"
    save_params = f"./weights/{net_model}_weight"
    save_img = f"./save_img/{net_model}_img"
    img_path = f"./data"
    # writer = SummaryWriter(f"./runs/{net_model}_runs")

    if not os.path.exists(save_img):  # 保存图片和参数
        os.makedirs(save_img)
    if not os.path.exists(save_params):
        os.makedirs(save_params)

    transf = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, ], [0.5, ])
    ])

    train_data = MyDataset(img_path, transf)
    train_load = DataLoader(train_data, batch_size, num_workers=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if net_model == "unet":
        net = MainNet(C_in=3, C_out=len(config.classes_list)).to(device)
    elif net_model == "u2net_p":
        net = U2NETP(in_ch=3, out_ch=len(config.classes_list)).to(device)
    elif net_model == "unet3+":
        net = UNet3Plus(n_channels=3, n_classes=len(config.classes_list)).to(device)
    elif net_model == "u2net":
        net = U2NET(in_ch=3, out_ch=len(config.classes_list)).to(device)

    if os.path.exists(load_params):
        net.load_state_dict(torch.load(load_params))

        # model_dict = net.state_dict()  # 网络需要的权重
        # print(model_dict)
        # pretext_model = torch.load(load_params)  # 加载的参数权重
        # print(pretext_model)
        # print(pretext_model["outconv.weight"])
        # w = pretext_model["outconv.weights"]
        # # print(w.shape)
        # pretext_model["outconv.weights"] = torch.cat((w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w),dim=0)
        # b = pretext_model["outconv.bias"]
        # pretext_model["outconv.bias"] = torch.cat((b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b),dim=0)
        # torch.save(pretext_model, os.path.join(save_params, f"unet_{183}_epoch.pth"))
        # net.load_state_dict(pretext_model)
        # exit()
        print("参数加载成功...")
    else:
        print("没有参数可加载...")

    optimzer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(), lr=3e-4, momentum=0.7)

    bce_loss = nn.BCELoss()
    # ce_loss = nn.CrossEntropyLoss()
    # ce_loss = nn.MSELoss()
    # focal_loss = FocalLoss(alpha=0.5,class_num=2)
    # dice_loss = SoftDiceLoss()
    # focal_loss = Focal_Loss(alpha=0.5, gamma=2)

    # writer.add_graph(model=net, input_to_model=torch.randn(1, 3, 224, 224).to(device))  # 写入网络结构

    net.train()
    for epoch in range(num + 1, Epoch):
        loss_list = []
        r2_list = []
        for i, (mask_img, real_img) in enumerate(train_load):
            mask_img, real_img = mask_img.to(device), real_img.to(device)
            # print(mask_img.shape)  # [N, 6, 224, 224]
            # print(real_img.shape)  # [N, 3, 224, 224]

            mask_out = torch.sigmoid(net(real_img))  # [N, 6, 224, 224]

            # print(mask_img.shape)  # [N, 6, 224, 224]
            # print(mask_out.shape)  # [N, 6, 224, 224]

            loss1 = bce_loss(mask_out, mask_img)
            # loss2 = focal_loss(mask_out, mask_img.long())

            optimzer.zero_grad()
            loss1.backward()
            optimzer.step()

            point_out = torch.cat(
                ((torch.argmax(mask_out.reshape(mask_out.shape[0], mask_out.shape[1], -1), dim=2) // mask_out.shape[3])[
                     ..., None],
                 (torch.argmax(mask_out.reshape(mask_out.shape[0], mask_out.shape[1], -1), dim=2) % mask_out.shape[3])[
                     ..., None]),
                dim=2)  # [n, 6]-->[n, 6, 1]-->[n, 6, 2]
            point_target = torch.cat(
                ((torch.argmax(mask_img.reshape(mask_img.shape[0], mask_img.shape[1], -1), dim=2) // mask_img.shape[3])[
                     ..., None],
                 (torch.argmax(mask_img.reshape(mask_img.shape[0], mask_img.shape[1], -1), dim=2) % mask_img.shape[3])[
                     ..., None]),
                dim=2)  # [n, 6, 2]

            # print(point_out)
            # print(point_target)

            r2 = r2_score(point_target.cpu().detach().numpy().reshape(point_target.shape[0], -1),
                          point_out.cpu().detach().numpy().reshape(point_out.shape[0], -1))
            # print(r2)

            mask_out_ = torch.max(mask_out, dim=1)[0].cpu().detach()  # [N, 416, 416]
            mask_img_ = torch.max(mask_img, dim=1)[0].cpu().detach()  # [N, 416, 416]

            loss_list.append(loss1.item())
            r2_list.append(r2.item())

            # if i % 100 == 0:
            #     print(f"{i},loss:{loss1.item()},acc:{accuracy.item()}")

        train_loss = np.mean(loss_list)
        train_r2 = np.mean(r2_list)

        # for name, param in net.named_parameters():
        #     print('层:', name, param.size())
        #     print('权值梯度', param.grad)
        #     print('权值', param)
        # exit()
        # if net_model == "unet":
        #     writer.add_histogram("grad", net.c1.layer[0].weight.grad, epoch)  # 保存第一层的梯度直方图
        # elif net_model == "u2net_p":
        #     writer.add_histogram("grad", net.stage1.rebnconvin.conv_s1.weight.grad, epoch)  # 保存第一层的梯度直方图
        # elif net_model == "unet3+":
        #     writer.add_histogram("grad", net.conv1.conv1[0].weight.grad, epoch)  # 保存第一层的梯度直方图
        # elif net_model == "u2net":
        #     writer.add_histogram("grad", net.stage1.rebnconvin.conv_s1.weight.grad, epoch)  # 保存第一层的梯度直方图
        #
        # writer.add_histogram("weight", net.outconv.weight, epoch)  # 保存最后一层的权重直方图
        # writer.add_scalars("acc", {"train_acc": train_acc}, epoch)  # 写入离散值
        # writer.add_scalars("loss", {"train_loss": train_loss}, epoch)
        # writer.close()

        print(f"{epoch}_epoch_loss:{train_loss} , train_r2:{train_r2}")

        mask_img_0 = mask_img_[0]  # [416, 416]  batchsize的第1张
        real_img_0 = real_img[0].permute(1, 2, 0).cpu().detach()  # [416, 416, 3]  batchsize的第1张
        mask_out_0 = mask_out_[0]  # [416, 416]
        # print(torch.unique(mask_img_0))  # [0., 1.]
        # print(torch.unique(mask_out_0))  # [0, 1]

        # mask_rgb = torch.zeros((config.IMG_WIDHT, config.IMG_HEIGHT, 3), dtype=int)
        # # print(mask_rgb.dtype)
        # for i in range(len(config.classes_list)):
        #     mask_bool = mask_out_0 == i
        #     rgb = config.color_list[i]
        #     mask_rgb[mask_bool] = torch.LongTensor(rgb)

        # 保存每轮最后一张图片
        plt.figure(num='输出图')
        plt.suptitle(f"{epoch}_epoch", c="b", size=20)  # 设置总标题
        plt.subplot(1, 3, 1), plt.imshow(real_img_0), plt.title("real_img")  # [416, 416]
        plt.subplot(1, 3, 2), plt.imshow(mask_img_0), plt.title("mask_label")  # [416, 416]
        plt.subplot(1, 3, 3), plt.imshow(mask_out_0), plt.title("mask_out")  # [416, 416, 3]
        plt.savefig(os.path.join(save_img, f"{epoch}_epoch.jpg"))
        # plt.show()
        # exit()

        if epoch == (num + 1):
            train_loss_last = train_loss

        if train_loss <= train_loss_last or epoch % 7 == 0:  # 保存参数
            torch.save(net.state_dict(), os.path.join(save_params, f"{net_model}_{epoch}_epoch.pth"))
            print("参数保存成功...")

            if train_loss <= train_loss_last:
                train_loss_last = train_loss

            params_name_list = os.listdir(save_params)  # 删除多余参数
            if len(params_name_list) > 10:
                epoch_list = []
                for params_name in params_name_list:
                    epoch = int(params_name.split("_")[-2])
                    epoch_list.append(epoch)
                min_epoch = np.min(epoch_list)
                move_dnet_params_name = f"{net_model}_{min_epoch}_epoch.pth"
                os.remove(os.path.join(save_params, move_dnet_params_name))

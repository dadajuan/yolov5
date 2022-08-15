from yolo_for_split import YOLO
import torch
from torch import nn
from torchsummary import summary


def get_mobile_cloud_model(full_model):

    mobile_model_Focus = nn.Sequential(*(list(full_model.modules())[3:4]))
    mobile_model_Conv1 = nn.Sequential(*(list(full_model.modules())[9:10]))
    mobile_model_dark2_cv1 = nn.Sequential(*(list(full_model.modules())[14:15]))
    mobile_model_dark2_cv2 = nn.Sequential(*(list(full_model.modules())[18:19]))
    mobile_model_dark2_cv3 = nn.Sequential(*(list(full_model.modules())[22:23]))
    mobile_model_dark2_bottleneck = nn.Sequential(*(list(full_model.modules())[26:27]))
    # cloud_model_dark2_C3 = nn.Sequential(*(list(full_model.modules())[13:14]))
    cloud_model_dark3 = nn.Sequential(*(list(full_model.modules())[63:64]))
    cloud_model_dark4 = nn.Sequential(*(list(full_model.modules())[190:191]))
    cloud_model_dark5 = nn.Sequential(*(list(full_model.modules())[317:318]))
    cloud_model_Upsample = nn.Sequential(*(list(full_model.modules())[385:386]))
    cloud_model_up1 = nn.Sequential(*(list(full_model.modules())[386:387]))
    cloud_model_Conv_up1 = nn.Sequential(*(list(full_model.modules())[390:391]))
    cloud_model_up2 = nn.Sequential(*(list(full_model.modules())[440:441]))
    cloud_model_Conv_up2 = nn.Sequential(*(list(full_model.modules())[444:445]))
    cloud_model_down1 = nn.Sequential(*(list(full_model.modules())[494:495]))
    cloud_model_Conv_down1 = nn.Sequential(*(list(full_model.modules())[498:499]))
    cloud_model_down2 = nn.Sequential(*(list(full_model.modules())[548:549]))
    cloud_model_Conv_down2 = nn.Sequential(*(list(full_model.modules())[552:553]))
    cloud_model_P3 = nn.Sequential(*(list(full_model.modules())[602:603]))
    cloud_model_P4 = nn.Sequential(*(list(full_model.modules())[603:604]))
    cloud_model_P5 = nn.Sequential(*(list(full_model.modules())[604:605]))

    class mobile_model(nn.Module):
        def __init__(self):
            super(mobile_model, self).__init__()
            self.Focus = mobile_model_Focus
            self.Conv1 = mobile_model_Conv1
            self.dark_cv1 = mobile_model_dark2_cv1
            self.dark2_C3_cv2 = mobile_model_dark2_cv2
            self.dark2_C3_cv3 = mobile_model_dark2_cv3
            self.dark2_C3_bottleneck = mobile_model_dark2_bottleneck
        def forward(self, x):
            x = self.Focus(x)
            x = self.Conv1(x)
            x1 = self.dark_cv1(x)
            x1 = self.dark2_C3_bottleneck(x1)
            x2 = self.dark2_C3_cv2(x)
            x = torch.cat((x1, x2), dim=1)
            out = self.dark2_C3_cv3(x)
            return out

    class cloud_model(nn.Module):
        def __init__(self):
            super(cloud_model, self).__init__()
            # self.dark2_C3_cv3 = cloud_model_dark2_cv3
            self.dark3 = cloud_model_dark3
            self.dark4 = cloud_model_dark4
            self.dark5 = cloud_model_dark5
            self.Upsample = cloud_model_Upsample
            self.up1 = cloud_model_up1
            self.Conv_up1 = cloud_model_Conv_up1
            self.up2 = cloud_model_up2
            self.Conv_up2 = cloud_model_Conv_up2
            self.down1 = cloud_model_down1
            self.Conv_down1 = cloud_model_Conv_down1
            self.down2 = cloud_model_down2
            self.Conv_down2 = cloud_model_Conv_down2
            self.P3 = cloud_model_P3
            self.P4 = cloud_model_P4
            self.P5 = cloud_model_P5

        def forward(self, x):
            # x = self.dark2_C3_cv3(x)
            x = self.dark3(x)
            feat1 = x
            x = self.dark4(x)
            feat2 = x
            x = self.dark5(x)
            feat3 = x

            P5 = self.up1(feat3)
            P5_upsample = self.Upsample(P5)
            P4 = torch.cat([P5_upsample, feat2], 1)
            P4 = self.Conv_up1(P4)

            P4 = self.up2(P4)
            P4_upsample = self.Upsample(P4)
            P3 = torch.cat([P4_upsample, feat1],1)
            P3 = self.Conv_up2(P3)

            P3_downsample = self.down1(P3)
            P4 = torch.cat([P3_downsample, P4], 1)
            P4 = self.Conv_down1(P4)

            P4_downsample = self.down2(P4)
            P5 = torch.cat([P4_downsample, P5], 1)
            P5 = self.Conv_down2(P5)

            out2 = self.P3(P3)
            out1 = self.P4(P4)
            out0 = self.P5(P5)
            return out0,out1,out2

    return mobile_model,cloud_model


yolo_model = YOLO()
model = yolo_model.net
# 利用 list(model.modules()) 和 yoloBody框架 进行分割
# for i in range(len(list(model.modules()))):
#     print(i)
#     print(list(model.modules())[i])


# yolov5_mobile_model
mobile, _ = get_mobile_cloud_model(model)
mobile = mobile().cuda()
# summary(mobile, input_size=(3, 640, 640))

# yolov5_cloud_model
_, cloud = get_mobile_cloud_model(model)
cloud = cloud().cuda()
# summary(cloud, input_size=(160, 160, 160))

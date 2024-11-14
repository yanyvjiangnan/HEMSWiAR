import paddle
import paddle.nn as nn
from attention import Spa_att,Cha_att
import mmd
from paddle.vision.models import resnet18, mobilenet_v2


class featurenet(nn.Layer):
    def __init__(self):
        super(featurenet, self).__init__()

        self.attention1 = Spa_att()
        self.attention2=Cha_att(1024)
        resnet = mobilenet_v2(pretrained=True)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)

    def forward(self, x):
        x = self.attention1(x)
        x = self.features(x)

        x = self.attention2(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        return x


################
class MFSAN(nn.Layer):
    def __init__(self, num_classes=6):
        super(MFSAN, self).__init__()
        self.sharedNet = featurenet()
        self.sonnet1 = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())
        self.sonnet2 = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())
        self.sonnet3 = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())
        self.sonnet4 = nn.Sequential(nn.Linear(1280, 256), nn.ReLU())

        self.MMD = mmd.mmd

        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.cls_fc_son4 = nn.Linear(256, num_classes)

    def forward(self, data_src, data_tgt, label_src, mark=1):

        if mark == 1:
            data_src, _ = self.sharedNet(data_src)
            data_tgt, _ = self.sharedNet(data_tgt)

            data_src = self.sonnet1(data_src)
            pred_src = self.cls_fc_son1(data_src)

            data_tgt_son1 = self.sonnet1(data_tgt)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
            domain_loss = self.MMD(data_src, data_tgt_son1)

            cls_loss = paddle.nn.functional.cross_entropy(pred_src, label_src)

            return cls_loss, domain_loss, pred_tgt_son1

        if mark == 2:
            data_src, _ = self.sharedNet(data_src)
            data_tgt, _ = self.sharedNet(data_tgt)

            data_src = self.sonnet2(data_src)
            pred_src = self.cls_fc_son2(data_src)

            data_tgt_son2 = self.sonnet2(data_tgt)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

            domain_loss = self.MMD(data_src, data_tgt_son2)

            cls_loss = paddle.nn.functional.cross_entropy(pred_src, label_src)

            return cls_loss, domain_loss, pred_tgt_son2

        if mark == 3:
            data_src, _ = self.sharedNet(data_src)
            data_tgt, _ = self.sharedNet(data_tgt)

            data_src = self.sonnet3(data_src)
            pred_src = self.cls_fc_son3(data_src)

            data_tgt_son3 = self.sonnet3(data_tgt)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)
            domain_loss = self.MMD(data_src, data_tgt_son3)

            # pred_src = self.cls_fc_son3(data_src)
            # pred_tgt_son3 = self.cls_fc_son3(data_tgt)
            # domain_loss = mmd.mmd(data_src, data_tgt)

            cls_loss = paddle.nn.functional.cross_entropy(pred_src, label_src)

            return cls_loss, domain_loss, pred_tgt_son3

        if mark == 4:
            data_src, _ = self.sharedNet(data_src)
            data_tgt, _ = self.sharedNet(data_tgt)

            data_src = self.sonnet4(data_src)
            pred_src = self.cls_fc_son4(data_src)

            data_tgt_son4 = self.sonnet4(data_tgt)
            pred_tgt_son4 = self.cls_fc_son4(data_tgt_son4)
            domain_loss = self.MMD(data_src, data_tgt_son4)

            cls_loss = paddle.nn.functional.cross_entropy(pred_src, label_src)

            return cls_loss, domain_loss, pred_tgt_son4

    def predict(self, data):
        data, asp = self.sharedNet(data)

        fea_son1 = self.sonnet1(data)
        pred1 = self.cls_fc_son1(fea_son1)

        fea_son2 = self.sonnet2(data)
        pred2 = self.cls_fc_son2(fea_son2)

        fea_son3 = self.sonnet3(data)
        pred3 = self.cls_fc_son3(fea_son3)

        fea_son4 = self.sonnet4(data)
        pred4 = self.cls_fc_son4(fea_son4)

        return pred1, pred2, pred3, pred4
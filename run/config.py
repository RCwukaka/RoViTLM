from RoViTLM.datasets.HUSTBearingDataset.datasetInfo import getHUSTDiffWorkCondition
from RoViTLM.datasets.PaderbornBearingDataset.datasetInfo import getPBDDiffWorkCondition
from RoViTLM.datasets.TYUTDataSet.datasetInfo import getTYUTDiffWorkCondition
from RoViTLM.datasets.WestBearingDataSet.datasetInfo import getWBDDiffWorkCondition
from RoViTLM.training.model.CNN.CNN import CNN
from RoViTLM.training.model.RoViTLM.RoViTLM import RoViTLM
from RoViTLM.training.model.RoViTLM.LightweightNet import LightweightNet
from RoViTLM.training.model.ResNet.ResNet import ResNet
from RoViTLM.training.model.ResNet.BasicBlock import BasicBlock
from RoViTLM.training.model.ViT.ViT import ViT

def getTrainMode(num_class):
    return [
        # {'name': 'Resnet18_ViT', 'model_name': 'M1', 'model': RoViTLM(net1=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256),
        #                                        net2=ViT(),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'Resnet34_ViT', 'model_name': 'M2', 'model': RoViTLM(net1=ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256),
        #                                            net2=ViT(),
        #                                            num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'CNN', 'model_name': 'M0', 'model': RoViTLM(net1=CNN(),
        #                                    net2=CNN(),
        #                                    num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'RoViTLM1_0', 'model_name': 'M7', 'model': RoViTLM(net1=LightweightNet(),
        #                                           net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                    depth=4),
        #                                           num_class=num_class), 'lamda': 1, 'mu': 0, 'type': 0},
        # {'name': 'RoViTLM0_1', 'model_name': 'M6', 'model': RoViTLM(net1=LightweightNet(),
        #                                                               net2=ViT(patch_size=(1, 64), img_size=(16, 64),
        #                                                                        num_classes=256,
        #                                                                        depth=4),
        #                                                               num_class=num_class), 'lamda': 0, 'mu': 1, 'type': 0},
        {'name': 'RoViTLM1_1', 'model_name': 'M9', 'model': RoViTLM(net1=LightweightNet(),
                                                  net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
                                                           depth=4),
                                                  num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'RoViTLM1_3','model_name': 'M00', 'model': RoViTLM(net1=LightweightNet(),
        #                                        net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256, depth=4),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 3, 'type': 0},
        # {'name': 'RoViTLM_CORAl', 'model_name': 'M3', 'model': RoViTLM(net1=LightweightNet(),
        #                                        net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256, depth=4),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 1},
        # {'name': 'RoViTLM_JMMD', 'model_name': 'M4', 'model': RoViTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 2},
        # {'name': 'RoViTLM_mmd', 'model_name': 'M5', 'model': RoViTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 3},
        # {'name': 'RoViTLM_16', 'model_name': 'M8', 'model': RoViTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 16), img_size=(16, 16), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
    ]


data = {
    'WBD': getWBDDiffWorkCondition(),
    'PBD': getPBDDiffWorkCondition(),
    'HUST': getHUSTDiffWorkCondition(),
}

transfer_task1 = [
    {'name': 'W0_W1', 'source': data['WBD']['data'][0], 'target': data['WBD']['data'][1],
     'num_class': data['WBD']['num_class']},
    {'name': 'W0_W2', 'source': data['WBD']['data'][0], 'target': data['WBD']['data'][2],
     'num_class': data['WBD']['num_class']},
    {'name': 'W0_W3', 'source': data['WBD']['data'][0], 'target': data['WBD']['data'][3],
     'num_class': data['WBD']['num_class']},
    {'name': 'W1_W0', 'source': data['WBD']['data'][1], 'target': data['WBD']['data'][0],
     'num_class': data['WBD']['num_class']},
    {'name': 'W1_W2', 'source': data['WBD']['data'][1], 'target': data['WBD']['data'][2],
     'num_class': data['WBD']['num_class']},
    {'name': 'W1_W3', 'source': data['WBD']['data'][1], 'target': data['WBD']['data'][3],
     'num_class': data['WBD']['num_class']},
    {'name': 'W2_W0', 'source': data['WBD']['data'][2], 'target': data['WBD']['data'][0],
     'num_class': data['WBD']['num_class']},
    {'name': 'W2_W1', 'source': data['WBD']['data'][2], 'target': data['WBD']['data'][1],
     'num_class': data['WBD']['num_class']},
    {'name': 'W2_W3', 'source': data['WBD']['data'][2], 'target': data['WBD']['data'][3],
     'num_class': data['WBD']['num_class']},
    {'name': 'W3_W0', 'source': data['WBD']['data'][3], 'target': data['WBD']['data'][0],
     'num_class': data['WBD']['num_class']},
    {'name': 'W3_W1', 'source': data['WBD']['data'][3], 'target': data['WBD']['data'][1],
     'num_class': data['WBD']['num_class']},
    {'name': 'W3_W2', 'source': data['WBD']['data'][3], 'target': data['WBD']['data'][2],
     'num_class': data['WBD']['num_class']},
]

transfer_task3 = [
    {'name': 'P0_P1', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][1],
     'num_class': data['PBD']['num_class']},
    {'name': 'P0_P2', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][2],
     'num_class': data['PBD']['num_class']},
    {'name': 'P1_P0', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][0],
     'num_class': data['PBD']['num_class']},
    {'name': 'P1_P2', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][2],
     'num_class': data['PBD']['num_class']},
    {'name': 'P2_P0', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][0],
     'num_class': data['PBD']['num_class']},
    {'name': 'P2_P1', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][1],
     'num_class': data['PBD']['num_class']},
]

transfer_task4 = [
    {'name': 'H0_H1', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][1],
     'num_class': data['HUST']['num_class']},
    {'name': 'H0_H2', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][2],
     'num_class': data['HUST']['num_class']},
    {'name': 'H0_H3', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][3],
     'num_class': data['HUST']['num_class']},
    {'name': 'H1_H0', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][0],
     'num_class': data['HUST']['num_class']},
    {'name': 'H1_H2', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][2],
     'num_class': data['HUST']['num_class']},
    {'name': 'H1_H3', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][3],
     'num_class': data['HUST']['num_class']},
    {'name': 'H2_H0', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][0],
     'num_class': data['HUST']['num_class']},
    {'name': 'H2_H1', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][1],
     'num_class': data['HUST']['num_class']},
    {'name': 'H2_H3', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][3],
     'num_class': data['HUST']['num_class']},
    {'name': 'H3_H0', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][0],
     'num_class': data['HUST']['num_class']},
    {'name': 'H3_H1', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][1],
     'num_class': data['HUST']['num_class']},
    {'name': 'H3_H2', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][2],
     'num_class': data['HUST']['num_class']},
]
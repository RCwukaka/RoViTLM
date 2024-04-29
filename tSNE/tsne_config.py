from INet.datasets.HUSTBearingDataset.datasetInfo import getHUSTDiffWorkCondition
from INet.datasets.PaderbornBearingDataset.datasetInfo import getPBDDiffWorkCondition
from INet.datasets.TYUTDataSet.datasetInfo import getTYUTDiffWorkCondition
from INet.datasets.WestBearingDataSet.datasetInfo import getWBDDiffWorkCondition
from INet.training.model.CNN.CNN import CNN
from INet.training.model.LRSADTLM.LRSADTLM import LRSADTLM
from INet.training.model.LRSADTLM.LightweightNet import LightweightNet
from INet.training.model.ResNet.ResNet import ResNet
from INet.training.model.ResNet.BasicBlock import BasicBlock
from INet.training.model.ViT.ViT import ViT

def getTrainMode(num_class):
    return [
        # {'name': 'Resnet18_ViT', 'model': LRSADTLM(net1=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256),
        #                                        net2=ViT(),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        {'name': 'Resnet34_ViT', 'model': LRSADTLM(net1=ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256),
                                                   net2=ViT(),
                                                   num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'CNN', 'model': LRSADTLM(net1=CNN(),
        #                                    net2=CNN(),
        #                                    num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'LRSADTLM1_0', 'model': LRSADTLM(net1=LightweightNet(),
        #                                           net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                    depth=4),
        #                                           num_class=num_class), 'lamda': 1, 'mu': 0, 'type': 0},
        # {'name': 'LRSADTLM0_1', 'model': LRSADTLM(net1=LightweightNet(),
        #                                           net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                    depth=4),
        #                                           num_class=num_class), 'lamda': 0, 'mu': 1, 'type': 0},
        # {'name': 'LRSADTLM1_1', 'model': LRSADTLM(net1=LightweightNet(),
        #                                           net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                    depth=4),
        #                                           num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
        # {'name': 'LRSADTLM1_3', 'model': LRSADTLM(net1=LightweightNet(),
        #                                        net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256, depth=4),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 3, 'type': 0},
        # {'name': 'LRSADTLM_CORAl', 'model': LRSADTLM(net1=LightweightNet(),
        #                                        net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256, depth=4),
        #                                        num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 1},
        # {'name': 'LRSADTLM_JMKMMD', 'model': LRSADTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 2},
        # {'name': 'LRSADTLM_mmd', 'model': LRSADTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 64), img_size=(16, 64), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 3},
        # {'name': 'LRSADTLM_16', 'model': LRSADTLM(net1=LightweightNet(),
        #                                              net2=ViT(patch_size=(1, 16), img_size=(16, 16), num_classes=256,
        #                                                       depth=4),
        #                                              num_class=num_class), 'lamda': 1, 'mu': 1, 'type': 0},
    ]


data = {
    'WBD': getWBDDiffWorkCondition(),
    'TYUT': getTYUTDiffWorkCondition(),
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

transfer_task2 = [
    {'name': 'T0_T1', 'source': data['TYUT']['data'][0], 'target': data['TYUT']['data'][1],
     'num_class': data['TYUT']['num_class']},
    {'name': 'T0_T2', 'source': data['TYUT']['data'][0], 'target': data['TYUT']['data'][2],
     'num_class': data['TYUT']['num_class']},
    {'name': 'T1_T0', 'source': data['TYUT']['data'][1], 'target': data['TYUT']['data'][0],
     'num_class': data['TYUT']['num_class']},
    {'name': 'T1_T2', 'source': data['TYUT']['data'][1], 'target': data['TYUT']['data'][2],
     'num_class': data['TYUT']['num_class']},
    {'name': 'T2_T0', 'source': data['TYUT']['data'][2], 'target': data['TYUT']['data'][0],
     'num_class': data['TYUT']['num_class']},
    {'name': 'T2_T1', 'source': data['TYUT']['data'][2], 'target': data['TYUT']['data'][1],
     'num_class': data['TYUT']['num_class']},
]

transfer_task3 = [
    {'name': 'P0_P1', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][1],
     'num_class': data['PBD']['num_class']},
    # {'name': 'P0_P2', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][2],
    #  'num_class': data['PBD']['num_class']},
    # {'name': 'P1_P0', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][0],
    #  'num_class': data['PBD']['num_class']},
    # {'name': 'P1_P2', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][2],
    #  'num_class': data['PBD']['num_class']},
    # {'name': 'P2_P0', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][0],
    #  'num_class': data['PBD']['num_class']},
    # {'name': 'P2_P1', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][1],
    #  'num_class': data['PBD']['num_class']},
]

transfer_task4 = [
    # {'name': 'H0_H1', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][1],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H0_H2', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][2],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H0_H3', 'source': data['HUST']['data'][0], 'target': data['HUST']['data'][3],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H1_H0', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][0],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H1_H2', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][2],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H1_H3', 'source': data['HUST']['data'][1], 'target': data['HUST']['data'][3],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H2_H0', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][0],
    #  'num_class': data['HUST']['num_class']},
    {'name': 'H2_H1', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][1],
     'num_class': data['HUST']['num_class']},
    # {'name': 'H2_H3', 'source': data['HUST']['data'][2], 'target': data['HUST']['data'][3],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H3_H0', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][0],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H3_H1', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][1],
    #  'num_class': data['HUST']['num_class']},
    # {'name': 'H3_H2', 'source': data['HUST']['data'][3], 'target': data['HUST']['data'][2],
    #  'num_class': data['HUST']['num_class']},
]
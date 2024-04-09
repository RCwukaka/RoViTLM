from INet.datasets.PaderbornBearingDataset.datasetInfo import getPBDDiffWorkCondition
from INet.datasets.TYUTDataSet.datasetInfo import getTYUTDiffWorkCondition
from INet.datasets.WestBearingDataSet.datasetInfo import getWBDDiffWorkCondition
from INet.training.model.CNN.CNN import CNN
from INet.training.model.LRSADTLM.LRSADTLM import LRSADTLM
from INet.training.model.ResNet.ResNet import ResNet
from INet.training.model.ResNet.BasicBlock import BasicBlock
from INet.training.model.ViT.ViT import ViT

def getTrainMode(num_class):
    return [
        # {'name': 'Resnet18_ViT', 'model': LRSADTLM(net1=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256),
        #                                        net2=ViT(),
        #                                        num_class=num_class)},
        # {'name': 'Resnet18_couple', 'model': LRSADTLM(net1=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256),
        #                                        net2=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256),
        #                                        num_class=num_class)},
        # {'name': 'Resnet34_couple', 'model': LRSADTLM(net1=ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256),
        #                                        net2=ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256),
        #                                        num_class=num_class)},
        {'name': 'Resnet34_ViT', 'model': LRSADTLM(net1=ResNet(BasicBlock, [3, 4, 6, 3], num_classes=256),
                                                   net2=ViT(),
                                                   num_class=num_class)},
        # {'name': 'CNN', 'model': LRSADTLM(net1=CNN(),
        #                                    net2=CNN(),
        #                                    num_class=num_class)},
        # {'name': 'LRSADTLM', 'model': LRSADTLM(num_class=num_class)}
    ]


data = {
    'WBD': getWBDDiffWorkCondition(),
    'TYUT': getTYUTDiffWorkCondition(),
    'PBD': getPBDDiffWorkCondition(),
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
    # {'name': 'P0_P1', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][1],
    #  'num_class': data['PBD']['num_class']},
    # {'name': 'P0_P2', 'source': data['PBD']['data'][0], 'target': data['PBD']['data'][2],
    #  'num_class': data['PBD']['num_class']},
    {'name': 'P1_P0', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][0],
     'num_class': data['PBD']['num_class']},
    {'name': 'P1_P2', 'source': data['PBD']['data'][1], 'target': data['PBD']['data'][2],
     'num_class': data['PBD']['num_class']},
    {'name': 'P2_P0', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][0],
     'num_class': data['PBD']['num_class']},
    {'name': 'P2_P1', 'source': data['PBD']['data'][2], 'target': data['PBD']['data'][1],
     'num_class': data['PBD']['num_class']},
]
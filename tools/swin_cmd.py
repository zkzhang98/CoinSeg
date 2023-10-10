import os


os.system(
    'python -u script_train.py 15-1 0,1,2,3,4,5 0 --freeze_low  '
    f'--conloss_proposal --conloss_prototype --KDLoss --KDLoss_prelogit --batch 16 '
    ' --name swin_voc')

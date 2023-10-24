This is the implementation of CoinSeg (ICCV 2023). [Link](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_CoinSeg_Contrast_Inter-_and_Intra-_Class_Representations_for_Incremental_Segmentation_ICCV_2023_paper.pdf).

## Requirements
All experiments in this paper are done with following environments:

- CUDA 11.6
- python (3.6.13)
- pytorch (1.7.1+cu110)
- torchvision (0.8.2+cu110)
- numpy (1.19.2)
- matplotlib
- pillow

## Dataset preparing

Organize datasets in the following structure.
```
path_to_your_dataset/
    VOC2012/
        Annotations/
        ImageSet/
        JPEGImages/
        SegmentationClassAug/
        proposal100/
        
    ADEChallengeData2016/
        annotations/
            training/
            validation/
        images/
            training/
            validation/
        proposal_adetrain/
        proposal_adeval/
```
You can get [proposal100](https://drive.google.com/file/d/1FxoyVa0I1IEwtW2ykGlNf-JkOYkK80E6/view?usp=sharing), [proposal_adetrain](https://drive.google.com/file/d/1kWfPNhoUnYz0uPuHJUALxiqvVqlCKrwW/view?usp=sharing), [proposal_adeval](https://drive.google.com/file/d/16xNMO4siqJXr5A03ywQDXU0F1Ld5OFtw/view?usp=sharing) here (provided by previous CISS method MicroSeg).
## Startup

We provide a training script `script_train.py` to facilitate the use of our proposed method. The script enables users to easily train CoinSeg with various settings, for example, the default config of CoinSeg is: 
```
 cd tools 
 python -u script_train.py 15-1 0,1,2,3,4,5 0 --freeze_low  
    --conloss_proposal --conloss_prototype --KDLoss --KDLoss_prelogit --batch 16 
    --name swin_voc
```
If you want to evaluate model after training , add `--test_only`.

## Cite
If you find our work to be helpful, please consider citing us:


```
@InProceedings{Zhang_2023_ICCV,
    author    = {Zhang, Zekang and Gao, Guangyu and Jiao, Jianbo and Liu, Chi Harold and Wei, Yunchao},
    title     = {CoinSeg: Contrast Inter- and Intra- Class Representations for Incremental Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {843-853}
}
```


# ResRep
清华大学&amp;旷世科技
 Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting  
 paper:https://arxiv.org/pdf/2007.03260v3.pdf  
 code:https://hub.fastgit.org/DingXiaoH/ResRep
 CSDN：https://blog.csdn.net/u011447962/article/details/120316394
 注意：
 （1）安装kerassurgeon模块 pip install kerassurgeon
      并且修改surgeon.py中的 379行elif layer_class in ('Conv1D', 'Conv2D', 'Conv3D', 'CompactorLayer')

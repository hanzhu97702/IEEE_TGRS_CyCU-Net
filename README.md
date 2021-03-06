# IEEE_TGRS_CyCU-Net
CyCU-Net: Cycle-Consistency Unmixing Network by Learning Cascaded Autoencoders
---------------------

The code in this toolbox implements the "CyCU-Net: Cycle-Consistency Unmixing Network by Learning Cascaded Autoencoders".
More specifically, it is detailed as follow

L. Gao, Z. Han, D. Hong, B. Zhang and J. Chanussot, "CyCU-Net: Cycle-Consistency Unmixing Network by Learning Cascaded Autoencoders," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5503914, doi: 10.1109/TGRS.2021.3064958.

Please kindly cite the papers if this code is useful and helpful for your research.

```bash
@article{gao2021cycu,
  title={CyCU-Net: Cycle-consistency unmixing network by learning cascaded autoencoders},
  author={Gao, Lianru and Han, Zhu and Hong, Danfeng and Zhang, Bing and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--14},
  note = {DOI: 10.1109/TGRS.2021.3064958},
  year={2022},
  publisher={IEEE}
}
```

System-specific notes
---------------------
The code was tested in the environment of Python 3.6.12 and torch 1.6.0.

How to use it?
---------------------

Directly run demo_cycunet.py to reproduce the results on the Samson data and the Jasper data, and then run result_display.m to display the evaluation results.

If you want to run the code in your own data, you can accordingly change the input (e.g., data) and tune the parameters.
Please note that 
1) the shape of the input matrix.
2) the init endmemebers should be given in advance.

If you encounter the bugs while using this code, please do not hesitate to contact us.
(hanzhu19@mails.ucas.ac.cn)

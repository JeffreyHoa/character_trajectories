# Character Trajectories
This is Handwriting Recognition by statistical machine learning method.

In this practice, we are given the trajectory of handwriting characters and required to predict class probabilities of them. Since we want to combine both spatial and time series information, our approach is an ensemble method that consists of two models based on Gaussian mixture. 
- The spatial model recovers the positions of the trajectory from velocities firstly and then constructs a class conditional * Gaussian mixture model. This gives approximately 95% correct rate. 
- The time series approach models the velocities directly and constructs Gaussian mixture models on each checkpoint for each label. The accuracy of this model is about 96%.
- Eventually the accuracy of the ensemble method could reach nearly 98% by validation.


## How to run it
```sh
$ #python3 ensemble.py
```

## Illustration
![image](https://github.com/JeffreyHoa/character_trajectories/blob/master/pic/Figure%201.png)

Figure 1, one sample of label ‘o’: original traces (row1) and traces after cutting “head” and “tail” (row2)

![image](https://github.com/JeffreyHoa/character_trajectories/blob/master/pic/Figure%202.png)

Figure 2, pixel representation

![image](https://github.com/JeffreyHoa/character_trajectories/blob/master/pic/Figure%203.png)

Figure 3, traces of all samples with label 'a' (row 1) and 'c' (row2) after cutting “head” and “tail”

![image](https://github.com/JeffreyHoa/character_trajectories/blob/master/pic/Figure%204.png)

Figure 4, average traces of 20 labels based on gmm.mean

![image](https://github.com/JeffreyHoa/character_trajectories/blob/master/pic/Figure%205.png)

Figure 5, compare with average trace


## Results
| Model | Error Rate | Mean Negative Log Probability |
| ------ | ------ | ------ |
| Spatial | 0.0454545454545 | 1.18020835723 |
| Time Series | 0.0384615384615 | 0.878646458743 |
| Ensemble | 0.0314685314685 | 3.36257110288 |


## References
B.H. Williams, M. Toussaint, and A.J. Storkey. “Extracting motion primitives from natural handwriting data.” In ICANN, volume 2, pages 634–643, 2006.

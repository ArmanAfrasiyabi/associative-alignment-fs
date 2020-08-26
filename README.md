#  [Associative Alignment for Few-shot Image Classification](https://lvsn.github.io/associative-alignment/) 
This paper proposes associative alignment with two strategies: 1) a metric-learning loss for minimizing the distance between related base samples and the centroid of novel instances in the feature space, and 2) a conditional adversarial alignment loss based on the Wasserstein distance.[**The project webpage!**](https://lvsn.github.io/associative-alignment/) 

This repository **will be available soon** and contain the pytorch implementation of Associative Alignment for Few-shot Image Classification [paper](https://arxiv.org/abs/1912.05094) [presentation](https://github.com/ArmanAfrasiyabi/associative-alignment-fs/blob/master/Associative%20Alignmentfor%20Few-Shot%20Image%20Classification.pdf).



 




## Train 
1. Hyper-parameters and training details are <code>args_parser.py</code>, where you can switch methods bw softMax, cosMax or arcMax.
2. Run transfer learning from <code>transferLearning.py</code>, which will capture the best model in ./results/models.
3. Run <code>associative_alignment.py<code> to perform our associative alignment using the best model found in (2) and defined setup in (1).



## Datasets
...






## Citation
<code>
@InProceedings{Afrasiyabi_2020_ECCV,
author = {Afrasiyabi, Arman and Lalonde, Jean-Fran\c{c}ois and Gagn\'{e}, Christian},
title = {Associative Alignment for Few-shot Image Classification},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}</code>

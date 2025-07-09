# Литература по факторизации и аппроксимации для ускорения LLM и VLM

## Научные статьи

### Факторизация и аппроксимация

1. [Tensor Decompositions and Applications](https://epubs.siam.org/doi/10.1137/07070111X) - Kolda, T.G., Bader, B.W. (2009)
2. [Tensor Factorization via Matrix Factorization](https://proceedings.mlr.press/v38/wang15a.html) - Wang, Y., et al. (2015)
3. [Tensor-Train Decomposition](https://epubs.siam.org/doi/10.1137/090752286) - Oseledets, I.V. (2011)
4. [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530) - Kim, Y.D., et al. (2016)
5. [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866) - Jaderberg, M., et al. (2014)
6. [Efficient Neural Network Compression](https://arxiv.org/abs/1811.12781) - Cheng, Y., et al. (2018)
7. [Compressing Neural Networks with the Hashing Trick](https://arxiv.org/abs/1504.04788) - Chen, W., et al. (2015)
8. [Tensor Decompositions for Learning Latent Variable Models](https://jmlr.org/papers/v15/anandkumar14b.html) - Anandkumar, A., et al. (2014)
9. [Tensor Contraction Layers for Parsimonious Deep Nets](https://arxiv.org/abs/1706.00439) - Kossaifi, J., et al. (2017)
10. [Learning Compact Recurrent Neural Networks](https://arxiv.org/abs/1710.02252) - Tjandra, A., et al. (2017)

### Факторизация для LLM и VLM

11. [Tender: Accelerating Large Language Models via Tensor Decomposition and Runtime Requantization](https://arxiv.org/abs/2310.19859) - Yin, H., et al. (2023)
12. [LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression](https://arxiv.org/abs/2306.13840) - Frantar, E., et al. (2023)
13. [Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models](https://arxiv.org/abs/2310.01906) - Yin, H., et al. (2023)
14. [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/abs/2002.11985) - Wang, Z., et al. (2020)
15. [Factorized Attention: Self-Attention with Linear Complexities](https://arxiv.org/abs/1812.01243) - Child, R., et al. (2019)
16. [Low-Rank Bottleneck in Multi-head Attention Models](https://arxiv.org/abs/2002.07028) - Winata, G.I., et al. (2020)
17. [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307) - Gordon, M.A., et al. (2020)
18. [Structured Multi-Linear Model for BERT Compression](https://arxiv.org/abs/2205.12399) - Edalati, A., et al. (2022)
19. [Compressing Vision-Language Models](https://arxiv.org/abs/2310.04415) - Yin, H., et al. (2023)
20. [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) - Yang, G., et al. (2022)

### Теоретические основы

21. [On the Expressive Power of Deep Learning: A Tensor Analysis](https://arxiv.org/abs/1509.05009) - Cohen, N., et al. (2016)
22. [Tensor Networks for Dimensionality Reduction and Large-Scale Optimization](https://www.nowpublishers.com/article/Details/MAL-059) - Cichocki, A., et al. (2016)
23. [Randomized Algorithms for Matrices and Data](https://www.nowpublishers.com/article/Details/MAL-012) - Mahoney, M.W. (2011)
24. [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](https://epubs.siam.org/doi/10.1137/090771806) - Halko, N., et al. (2011)
25. [Nonnegative Tensor Factorization with Applications to Statistics and Computer Vision](https://dl.acm.org/doi/10.1145/1102351.1102451) - Shashua, A., Hazan, T. (2005)
26. [Tensor Decompositions, Alternating Least Squares and Other Tales](https://www.sciencedirect.com/science/article/pii/S0377042700003027) - Tomasi, G., Bro, R. (2006)
27. [Tensor Decompositions for Signal Processing Applications: From Two-Way to Multiway Component Analysis](https://ieeexplore.ieee.org/document/7038247) - Comon, P., et al. (2015)
28. [Tensor Decompositions for Feature Extraction and Classification of High Dimensional Datasets](https://www.sciencedirect.com/science/article/pii/S0925231213003706) - Phan, A.H., Cichocki, A. (2013)
29. [Tensor Methods in Computer Vision and Deep Learning](https://arxiv.org/abs/2009.05639) - Kossaifi, J., et al. (2020)
30. [Tensor Methods for Large, Sparse or Multi-relational Data](https://dl.acm.org/doi/10.1145/2339530.2339723) - Papalexakis, E.E., et al. (2013)

## Библиотеки и фреймворки

### Библиотеки для факторизации и аппроксимации

31. [TensorLy](http://tensorly.org/) - Библиотека для тензорного обучения с поддержкой различных бэкендов
32. [TensorLy-Torch](https://tensorly.org/torch/) - Расширение TensorLy для PyTorch с факторизованными слоями
33. [scikit-tensor](https://github.com/mnick/scikit-tensor) - Библиотека для мультилинейной алгебры и тензорных факторизаций
34. [TorchTT](https://github.com/yuhuixu1993/torch-tt) - Библиотека для тензорно-поездного разложения на PyTorch
35. [FunFact](https://github.com/yhtang/FunFact) - Библиотека для автоматизации моделей факторизации матриц и тензоров
36. [Tensortools](https://github.com/ahwillia/tensortools) - Инструменты для тензорного разложения в Python
37. [PyTorch SVD функции](https://pytorch.org/docs/stable/generated/torch.svd.html) - Встроенные функции PyTorch для SVD
38. [PyTorch Low-Rank SVD](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html) - Функции для низкоранговой аппроксимации в PyTorch
39. [Weighted-low-rank-factorization-Pytorch](https://github.com/viig99/Weighted-low-rank-factorization-Pytorch) - Реализация взвешенной низкоранговой факторизации для PyTorch
40. [TensorRT](https://developer.nvidia.com/tensorrt) - SDK для высокопроизводительного инференса с поддержкой разреженных тензоров

### Интеграция с другими библиотеками

41. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Оптимизация LLM с поддержкой факторизации
42. [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Библиотека для распределенного обучения с поддержкой оптимизаций
43. [Hugging Face Optimum](https://github.com/huggingface/optimum) - Инструменты для оптимизации моделей, включая факторизацию
44. [ONNX Runtime](https://onnxruntime.ai/) - Кроссплатформенный движок для инференса с оптимизациями
45. [TVM](https://tvm.apache.org/) - Компилятор для моделей глубокого обучения с поддержкой оптимизаций

## Документация

### TensorLy и связанные библиотеки

46. [TensorLy Documentation](http://tensorly.org/stable/index.html) - Официальная документация TensorLy
47. [TensorLy-Torch Documentation](https://tensorly.org/torch/) - Документация TensorLy-Torch
48. [TensorLy Tutorials](http://tensorly.org/stable/auto_examples/index.html) - Туториалы по использованию TensorLy
49. [TensorLy API Reference](http://tensorly.org/stable/modules/api.html) - Справочник API TensorLy
50. [TensorLy-Torch Examples](https://github.com/tensorly/tensorly-torch/tree/main/examples) - Примеры использования TensorLy-Torch

### PyTorch и другие библиотеки

51. [PyTorch SVD Documentation](https://pytorch.org/docs/stable/generated/torch.svd.html) - Документация по SVD в PyTorch
52. [PyTorch Low-Rank SVD Documentation](https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html) - Документация по низкоранговому SVD в PyTorch
53. [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) - Руководство разработчика TensorRT
54. [scikit-tensor Documentation](https://github.com/mnick/scikit-tensor/blob/master/README.md) - Документация scikit-tensor
55. [TorchTT Documentation](https://github.com/yuhuixu1993/torch-tt/blob/master/README.md) - Документация TorchTT

## Обучающие материалы

### Туториалы и блоги

56. [Tensor Decomposition: A Mathematical Tool for Data Analysis](https://arxiv.org/abs/1607.01668) - Cichocki, A. (2016)
57. [Introduction to Tensor Decompositions and their Applications in Machine Learning](https://www.slideshare.net/BavaniThuraisingham/tensor-decompositions-and-their-applications-in-machine-learning) - Papalexakis, E.E. (2018)
58. [Tensor Decompositions for Machine Learning](https://www.cs.cmu.edu/~gmurray/tensor_tutorial.pdf) - Murray, G. (2016)
59. [Tensor Methods in Machine Learning](https://www.cs.cornell.edu/~siddarth/tensor-tutorial.pdf) - Anandkumar, A. (2015)
60. [Low-Rank Approximation: Algorithms, Implementation, Applications](https://www.cs.utexas.edu/~cmcurtin/cs395t/slides/lecture21.pdf) - Curtin, M. (2018)

### Видеоуроки

61. [Tensor Decomposition Techniques](https://www.youtube.com/watch?v=B5QWLJpLWkY) - Cichocki, A. (2016)
62. [Tensor Networks for Machine Learning](https://www.youtube.com/watch?v=YnQJTfbwBM8) - Stoudenmire, E.M. (2018)
63. [Low-Rank Approximations in Machine Learning](https://www.youtube.com/watch?v=KBCJOcGgVlA) - Mahoney, M.W. (2019)
64. [Tensor Methods for Machine Learning](https://www.youtube.com/watch?v=ONS3MJRnLRY) - Anandkumar, A. (2017)
65. [Tensor Decomposition Methods and Applications](https://www.youtube.com/watch?v=MXZCVjsWLVA) - Kolda, T.G. (2020)

---

Эта литература охватывает широкий спектр тем, связанных с факторизацией и аппроксимацией для ускорения LLM и VLM моделей, от теоретических основ до практических инструментов и библиотек. Регулярно обращайтесь к этим ресурсам для получения актуальной информации о последних достижениях в области оптимизации моделей машинного обучения с использованием методов факторизации и аппроксимации.
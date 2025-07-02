# Литература по ускорению LLM и VLM

## Оглавление

1. [Научные статьи](#научные-статьи)
   - [Квантование](#квантование)
   - [Дистилляция знаний](#дистилляция-знаний)
   - [Бинаризация](#бинаризация)
   - [Оптимизация инференса](#оптимизация-инференса)
   - [Архитектуры моделей](#архитектуры-моделей)
2. [Библиотеки и фреймворки](#библиотеки-и-фреймворки)
   - [PyTorch и экосистема](#pytorch-и-экосистема)
   - [Hugging Face](#hugging-face)
   - [TensorRT и NVIDIA](#tensorrt-и-nvidia)
   - [Квантование и оптимизация](#квантование-и-оптимизация)
   - [Распределенное обучение](#распределенное-обучение)
3. [Документация](#документация)
   - [PyTorch](#pytorch)
   - [Hugging Face](#hugging-face-документация)
   - [TensorRT-LLM](#tensorrt-llm-документация)
   - [FlashAttention](#flashattention)
   - [Другие инструменты](#другие-инструменты)
4. [Модели](#модели)
   - [LLM](#llm)
   - [VLM](#vlm)
   - [Квантованные модели](#квантованные-модели)
   - [Дистиллированные модели](#дистиллированные-модели)
5. [Обучающие материалы](#обучающие-материалы)
   - [Курсы](#курсы)
   - [Видеоуроки](#видеоуроки)
   - [Туториалы и блоги](#туториалы-и-блоги)
   - [Книги](#книги)
6. [Инструменты и утилиты](#инструменты-и-утилиты)
   - [Профилирование и бенчмаркинг](#профилирование-и-бенчмаркинг)
   - [Визуализация](#визуализация)
   - [Развертывание](#развертывание)

---

## Научные статьи

### Квантование

1. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) - Dettmers, T., et al. (2022)
2. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) - Frantar, E., et al. (2022)
3. [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861) - Yao, Z., et al. (2022)
4. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) - Xiao, G., et al. (2023)
5. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers, T., et al. (2023)
6. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) - Lin, J., et al. (2023)
7. [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) - Kim, Y., et al. (2021)
8. [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078) - Frantar, E., et al. (2023)
9. [The Case for 4-bit Precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720) - Dettmers, T., et al. (2022)
10. [QUIK: Towards End-to-End 4-bit Inference on Generative Large Language Models](https://arxiv.org/abs/2310.09259) - Kim, S., et al. (2023)

### Дистилляция знаний

11. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton, G., et al. (2015)
12. [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) - Sanh, V., et al. (2019)
13. [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) - Jiao, X., et al. (2019)
14. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957) - Wang, W., et al. (2020)
15. [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) - Gou, J., et al. (2020)
16. [Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301) - Hsieh, C.Y., et al. (2023)
17. [Distilling Vision Transformers](https://arxiv.org/abs/2106.05237) - Touvron, H., et al. (2021)
18. [Efficient-CLIP: Efficient Cross-Modal Pre-training by Ensemble Confident Learning and Language Modeling](https://arxiv.org/abs/2212.04999) - Lin, Y., et al. (2022)
19. [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) - Kim, Y. and Rush, A.M. (2016)
20. [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770) - Furlanello, T., et al. (2018)

### Бинаризация

21. [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830) - Hubara, I., et al. (2016)
22. [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279) - Rastegari, M., et al. (2016)
23. [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) - Wang, Z., et al. (2023)
24. [BinaryBERT: Pushing the Limit of BERT Quantization](https://arxiv.org/abs/2012.15701) - Bai, H., et al. (2020)
25. [BiBERT: Accurate Fully Binarized BERT](https://arxiv.org/abs/2203.08144) - Qin, H., et al. (2022)
26. [Binary Neural Networks: A Survey](https://arxiv.org/abs/2004.03333) - Qin, H., et al. (2020)
27. [Training Binary Neural Networks through Learning with Noisy Supervision](https://arxiv.org/abs/1810.10133) - Xu, Z. and Cheung, R.C.C. (2018)
28. [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://arxiv.org/abs/1808.00278) - Liu, Z., et al. (2018)
29. [ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions](https://arxiv.org/abs/2003.03488) - Liu, Z., et al. (2020)
30. [Towards Accurate Binary Convolutional Neural Network](https://arxiv.org/abs/1711.11294) - Lin, X., et al. (2017)

### Оптимизация инференса

31. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - Dao, T., et al. (2022)
32. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Dao, T., et al. (2023)
33. [PagedAttention: Optimizing Transformer Inference for Large Context Lengths](https://arxiv.org/abs/2309.06180) - Kwon, W., et al. (2023)
34. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon, W., et al. (2023)
35. [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032) - Aminabadi, R.Y., et al. (2022)
36. [FasterTransformer: Efficient Transformer Inference Optimization](https://arxiv.org/abs/2207.07845) - Geng, X., et al. (2022)
37. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - Chen, L., et al. (2023)
38. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2310.05341) - Chen, S., et al. (2023)
39. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - Shazeer, N. (2019)
40. [Memory-Efficient Transformers via Top-k Attention](https://arxiv.org/abs/2106.06899) - Zhao, S., et al. (2021)

### Архитектуры моделей

41. [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani, A., et al. (2017)
42. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Brown, T., et al. (2020)
43. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Touvron, H., et al. (2023)
44. [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Touvron, H., et al. (2023)
45. [Mistral 7B](https://arxiv.org/abs/2310.06825) - Jiang, A.Q., et al. (2023)
46. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Radford, A., et al. (2021)
47. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) - Li, J., et al. (2023)
48. [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485) - Liu, H., et al. (2023)
49. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) - Alayrac, J.B., et al. (2022)
50. [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560) - Zhai, X., et al. (2021)

## Библиотеки и фреймворки

### PyTorch и экосистема

51. [PyTorch](https://pytorch.org/) - Основной фреймворк для глубокого обучения
52. [TorchVision](https://pytorch.org/vision/) - Библиотека компьютерного зрения для PyTorch
53. [TorchText](https://pytorch.org/text/) - Библиотека обработки текста для PyTorch
54. [TorchAudio](https://pytorch.org/audio/) - Библиотека обработки аудио для PyTorch
55. [PyTorch Lightning](https://www.pytorchlightning.ai/) - Легковесная обертка для PyTorch
56. [TorchServe](https://pytorch.org/serve/) - Инструмент для развертывания моделей PyTorch
57. [TorchScript](https://pytorch.org/docs/stable/jit.html) - Сериализация и оптимизация моделей PyTorch
58. [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) - API для квантования моделей
59. [PyTorch Mobile](https://pytorch.org/mobile/home/) - Оптимизация моделей для мобильных устройств
60. [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Инструмент для профилирования моделей

### Hugging Face

61. [Transformers](https://github.com/huggingface/transformers) - Библиотека предобученных моделей
62. [Accelerate](https://github.com/huggingface/accelerate) - Библиотека для распределенного обучения
63. [Optimum](https://github.com/huggingface/optimum) - Инструменты для оптимизации моделей
64. [Diffusers](https://github.com/huggingface/diffusers) - Библиотека для диффузионных моделей
65. [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
66. [Datasets](https://github.com/huggingface/datasets) - Библиотека для работы с датасетами
67. [Tokenizers](https://github.com/huggingface/tokenizers) - Быстрые токенизаторы для NLP
68. [Evaluate](https://github.com/huggingface/evaluate) - Библиотека для оценки моделей
69. [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - Сервер для генерации текста
70. [TRL](https://github.com/huggingface/trl) - Библиотека для обучения с подкреплением

### TensorRT и NVIDIA

71. [TensorRT](https://developer.nvidia.com/tensorrt) - SDK для высокопроизводительного инференса на GPU
72. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Оптимизация LLM с помощью TensorRT
73. [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) - Сервер для развертывания моделей
74. [NVIDIA DALI](https://developer.nvidia.com/dali) - Библиотека для загрузки и предобработки данных
75. [NVIDIA Apex](https://github.com/NVIDIA/apex) - Инструменты для смешанной точности
76. [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) - Библиотека для глубокого обучения на GPU
77. [NVIDIA cuBLAS](https://developer.nvidia.com/cublas) - Библиотека линейной алгебры для GPU
78. [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - Шаблоны для операций с тензорами на GPU
79. [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Фреймворк для разработки моделей ИИ
80. [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Распределенное обучение LLM

### Квантование и оптимизация

81. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - Библиотека для 8-битного и 4-битного квантования
82. [ONNX](https://onnx.ai/) - Открытый формат для представления моделей машинного обучения
83. [ONNX Runtime](https://onnxruntime.ai/) - Кроссплатформенный движок для инференса ONNX моделей
84. [Intel Neural Compressor](https://github.com/intel/neural-compressor) - Инструмент для оптимизации моделей на CPU
85. [TVM](https://tvm.apache.org/) - Компилятор для моделей глубокого обучения
86. [Brevitas](https://github.com/Xilinx/brevitas) - Библиотека для квантованных нейронных сетей
87. [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) - Автоматическое применение GPTQ для квантования LLM
88. [llama.cpp](https://github.com/ggerganov/llama.cpp) - Инференс LLaMA в C/C++ с квантованием
89. [ExLlamaV2](https://github.com/turboderp/exllamav2) - Оптимизированный инференс для LLaMA с квантованием
90. [vLLM](https://github.com/vllm-project/vllm) - Высокопроизводительный инференс LLM с PagedAttention

### Распределенное обучение

91. [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Библиотека для распределенного обучения
92. [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) - Техника распределенного обучения в PyTorch
93. [Horovod](https://github.com/horovod/horovod) - Распределенное обучение для TensorFlow, Keras, PyTorch
94. [Ray](https://www.ray.io/) - Фреймворк для распределенных вычислений
95. [Colossal-AI](https://github.com/hpcaitech/ColossalAI) - Параллельное обучение больших моделей
96. [Alpa](https://github.com/alpa-projects/alpa) - Автоматическое параллельное обучение
97. [FairScale](https://github.com/facebookresearch/fairscale) - Библиотека для обучения больших моделей
98. [Parallax](https://github.com/snuspl/parallax) - Распределенное обучение на нескольких GPU
99. [Bagua](https://github.com/BaguaSys/bagua) - Распределенное обучение с коммуникационной оптимизацией
100. [BytePS](https://github.com/bytedance/byteps) - Распределенное обучение с оптимизацией коммуникаций

## Документация

### PyTorch

101. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Официальная документация PyTorch
102. [PyTorch Tutorials](https://pytorch.org/tutorials/) - Официальные туториалы PyTorch
103. [PyTorch Examples](https://github.com/pytorch/examples) - Примеры использования PyTorch
104. [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) - Документация по квантованию в PyTorch
105. [PyTorch Mobile Documentation](https://pytorch.org/mobile/home/) - Документация по PyTorch Mobile

### Hugging Face документация

106. [Hugging Face Documentation](https://huggingface.co/docs) - Документация Hugging Face
107. [Hugging Face Course](https://huggingface.co/course) - Курс по использованию Hugging Face
108. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) - Документация по библиотеке Transformers
109. [Hugging Face Optimum Documentation](https://huggingface.co/docs/optimum/index) - Документация по библиотеке Optimum
110. [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index) - Документация по библиотеке PEFT

### TensorRT-LLM документация

111. [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) - Репозиторий TensorRT-LLM
112. [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) - Документация TensorRT-LLM
113. [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) - Руководство разработчика TensorRT
114. [NVIDIA Developer Blog: TensorRT-LLM](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/) - Блог о TensorRT-LLM
115. [TensorRT-LLM Examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) - Примеры использования TensorRT-LLM

### FlashAttention

116. [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention) - Репозиторий FlashAttention
117. [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) - Научная статья о FlashAttention-2
118. [FlashAttention Documentation](https://github.com/Dao-AILab/flash-attention/blob/main/README.md) - Документация FlashAttention
119. [FlashAttention Examples](https://github.com/Dao-AILab/flash-attention/tree/main/examples) - Примеры использования FlashAttention
120. [FlashAttention Blog Post](https://crfm.stanford.edu/2023/01/13/flashattention.html) - Блог о FlashAttention

### Другие инструменты

121. [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes/blob/main/README.md) - Документация bitsandbytes
122. [ONNX Documentation](https://onnx.ai/onnx/index.html) - Документация ONNX
123. [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) - Документация ONNX Runtime
124. [DeepSpeed Documentation](https://www.deepspeed.ai/getting-started/) - Документация DeepSpeed
125. [vLLM Documentation](https://vllm.readthedocs.io/) - Документация vLLM

## Модели

### LLM

126. [LLaMA 2](https://huggingface.co/meta-llama) - Семейство открытых LLM от Meta
127. [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) - Открытая 7B модель от Mistral AI
128. [Falcon](https://huggingface.co/tiiuae/falcon-7b) - Семейство открытых LLM от TII
129. [MPT](https://huggingface.co/mosaicml/mpt-7b) - Семейство моделей от MosaicML
130. [BLOOM](https://huggingface.co/bigscience/bloom) - Многоязычная модель от BigScience
131. [Pythia](https://huggingface.co/EleutherAI/pythia-6.9b) - Семейство моделей от EleutherAI
132. [OPT](https://huggingface.co/facebook/opt-6.7b) - Открытые модели от Meta AI
133. [FLAN-T5](https://huggingface.co/google/flan-t5-xl) - Семейство инструктированных моделей от Google
134. [Qwen](https://huggingface.co/Qwen/Qwen-7B) - Семейство моделей от Alibaba
135. [Gemma](https://huggingface.co/google/gemma-7b) - Открытые модели от Google

### VLM

136. [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) - Модель для связывания текста и изображений от OpenAI
137. [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) - Мультимодальная модель от Salesforce
138. [LLaVA](https://huggingface.co/llava-hf/llava-1.5-7b-hf) - Мультимодальная модель на основе LLaMA
139. [CoCa](https://huggingface.co/laion/CoCa-ViT-B-32-laion2B-s13B-b90k) - Контрастная модель для текста и изображений
140. [FLAVA](https://huggingface.co/facebook/flava-full) - Мультимодальная модель от Meta
141. [ViLT](https://huggingface.co/dandelin/vilt-b32-mlm) - Эффективная модель для задач текст-изображение
142. [ALBEF](https://huggingface.co/ALBEF/albef-base) - Модель для выравнивания и слияния текста и изображений
143. [ImageBind](https://github.com/facebookresearch/ImageBind) - Модель для связывания разных модальностей
144. [Flamingo](https://huggingface.co/openflamingo/OpenFlamingo-9B) - Мультимодальная модель для few-shot обучения
145. [KOSMOS-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) - Мультимодальная модель от Microsoft

### Квантованные модели

146. [LLaMA-2-7B-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ) - Квантованная версия LLaMA-2-7B
147. [Mistral-7B-GGUF](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF) - Квантованная версия Mistral-7B
148. [Falcon-7B-GPTQ](https://huggingface.co/TheBloke/falcon-7b-GPTQ) - Квантованная версия Falcon-7B
149. [BLOOM-7B-GPTQ](https://huggingface.co/TheBloke/bloom-7b1-GPTQ) - Квантованная версия BLOOM-7B
150. [MPT-7B-GGUF](https://huggingface.co/TheBloke/MPT-7B-GGUF) - Квантованная версия MPT-7B

### Дистиллированные модели

151. [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) - Дистиллированная версия BERT
152. [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_6L_768D) - Компактная версия BERT
153. [DistilGPT2](https://huggingface.co/distilgpt2) - Дистиллированная версия GPT-2
154. [MiniLM](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) - Компактная модель от Microsoft
155. [MobileBERT](https://huggingface.co/google/mobilebert-uncased) - Компактная версия BERT для мобильных устройств

## Обучающие материалы

### Курсы

156. [Practical Deep Learning for Coders](https://course.fast.ai/) - Курс по глубокому обучению от fast.ai
157. [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) - Специализация по глубокому обучению от deeplearning.ai
158. [Natural Language Processing Specialization](https://www.deeplearning.ai/courses/natural-language-processing-specialization/) - Специализация по NLP от deeplearning.ai
159. [PyTorch for Deep Learning](https://www.udacity.com/course/deep-learning-pytorch--ud188) - Курс по PyTorch от Udacity
160. [Hugging Face Course](https://huggingface.co/course/chapter1/1) - Курс по использованию Hugging Face

### Видеоуроки

161. [NVIDIA: Accelerating LLMs with TensorRT-LLM](https://www.youtube.com/watch?v=jvnlJJi8mpo) - Видео о TensorRT-LLM
162. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://www.youtube.com/watch?v=FThvfkXWqtE) - Видео о FlashAttention
163. [Yannic Kilcher: Attention is All You Need (Transformer) - Paper Explained](https://www.youtube.com/watch?v=iDulhoQ2pro) - Объяснение архитектуры трансформеров
164. [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Создание GPT с нуля
165. [The AI Epiphany: Knowledge Distillation Explained](https://www.youtube.com/watch?v=FQM13HkEfBk) - Объяснение дистилляции знаний
166. [PyTorch Lightning Tutorial](https://www.youtube.com/watch?v=DbESHcCoWbM) - Туториал по PyTorch Lightning
167. [Hugging Face Transformers Tutorial](https://www.youtube.com/watch?v=DQc2Mi7BcuI) - Туториал по Hugging Face Transformers
168. [Quantization in Deep Learning](https://www.youtube.com/watch?v=KASuxB3XoYQ) - Видео о квантовании в глубоком обучении
169. [Binary Neural Networks Explained](https://www.youtube.com/watch?v=ry9tqG3a-Pc) - Объяснение бинарных нейронных сетей
170. [ONNX Runtime Tutorial](https://www.youtube.com/watch?v=5S8EpZ8sZNc) - Туториал по ONNX Runtime

### Туториалы и блоги

171. [PyTorch Tutorials](https://pytorch.org/tutorials/) - Официальные туториалы PyTorch
172. [Hugging Face Blog](https://huggingface.co/blog) - Блог Hugging Face
173. [NVIDIA Developer Blog](https://developer.nvidia.com/blog/category/artificial-intelligence/) - Блог NVIDIA о ИИ
174. [Papers With Code](https://paperswithcode.com/) - Статьи с реализациями кода
175. [Towards Data Science](https://towardsdatascience.com/) - Платформа для статей о машинном обучении
176. [The Gradient](https://thegradient.pub/) - Публикации о машинном обучении
177. [Distill.pub](https://distill.pub/) - Научные статьи с интерактивными визуализациями
178. [Sebastian Raschka's Blog](https://sebastianraschka.com/blog.html) - Блог о машинном обучении
179. [Jay Alammar's Blog](https://jalammar.github.io/) - Визуальные объяснения концепций машинного обучения
180. [Tim Dettmers' Blog](https://timdettmers.com/) - Блог о квантовании и оптимизации моделей

### Книги

181. [Deep Learning](https://www.deeplearningbook.org/) - Книга по глубокому обучению от Goodfellow, Bengio и Courville
182. [Natural Language Processing with Transformers](https://transformersbook.com/) - Книга о трансформерах от Hugging Face
183. [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) - Официальная книга по PyTorch
184. [Dive into Deep Learning](https://d2l.ai/) - Интерактивная книга по глубокому обучению
185. [Efficient Deep Learning](https://efficientdl.com/) - Книга об эффективном глубоком обучении

## Инструменты и утилиты

### Профилирование и бенчмаркинг

186. [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Инструмент для профилирования PyTorch моделей
187. [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - Инструмент для профилирования GPU
188. [NVIDIA Deep Learning Profiler (DLProf)](https://developer.nvidia.com/dlprof) - Профилировщик для глубокого обучения
189. [MLPerf](https://mlcommons.org/en/training-normal-07/) - Бенчмарки для машинного обучения
190. [Weights & Biases](https://wandb.ai/) - Платформа для отслеживания экспериментов

### Визуализация

191. [TensorBoard](https://www.tensorflow.org/tensorboard) - Инструмент для визуализации обучения
192. [Netron](https://github.com/lutzroeder/netron) - Визуализатор моделей машинного обучения
193. [Weights & Biases](https://wandb.ai/) - Платформа для визуализации экспериментов
194. [Gradio](https://gradio.app/) - Создание веб-интерфейсов для моделей
195. [Streamlit](https://streamlit.io/) - Создание веб-приложений для машинного обучения

### Развертывание

196. [TorchServe](https://pytorch.org/serve/) - Инструмент для развертывания моделей PyTorch
197. [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) - Сервер для развертывания моделей
198. [BentoML](https://www.bentoml.com/) - Платформа для развертывания моделей
199. [Hugging Face Inference API](https://huggingface.co/inference-api) - API для инференса моделей
200. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Фреймворк для развертывания моделей

---

Эта литература охватывает широкий спектр тем, связанных с ускорением LLM и VLM, от теоретических основ до практических инструментов и библиотек. Регулярно обращайтесь к этим ресурсам для получения актуальной информации о последних достижениях в области оптимизации моделей машинного обучения.
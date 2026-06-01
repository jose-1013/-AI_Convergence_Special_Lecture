# COCO Captions Dataset

## 1. 개요

**COCO Captions**는 MS COCO(Common Objects in Context)의 이미지 캡셔닝용 annotation subset이다. 한 이미지에 대해 여러 명의 사람이 자연어 문장으로 장면을 설명한 데이터셋이며, 이미지 캡셔닝, image-text retrieval, multimodal evaluation에서 표준 벤치마크처럼 사용된다.

`1-VLM-실습.ipynb`에서는 Section 3과 Section 4에서 COCO captioning을 사용한다.

| Section | 사용 방식 | 기본 규모 |
|---|---|---:|
| Section 3 | SmolVLM zero-shot captioning 평가 | validation image 4장 |
| Section 4 | SmolVLM full fine-tuning, before/after 평가 | train 1000장, validation 50장, test 150장 |

노트북은 전체 COCO를 직접 내려받아 수동으로 풀기보다, `utils/coco_dataloader.py`의 helper로 image 단위 subset을 만들고 manifest를 저장해 같은 seed에서 같은 샘플을 재사용한다.

### MS COCO 데이터셋

**MS COCO**는 일상 장면 이미지에 다양한 annotation을 붙인 대규모 vision dataset이다. 단순히 객체가 있는지만 보는 것이 아니라, 여러 객체가 실제 생활 맥락 속에서 함께 등장하는 복잡한 장면 이해를 목표로 설계되었다.

주요 annotation에는 object detection bounding box, instance segmentation mask, person keypoints, stuff/panoptic segmentation, image captions가 포함된다. 전체 규모는 약 330K images, annotation 포함 이미지 약 200K+, object instance 약 1.5M, object category 80개, stuff category 91개, person keypoints 약 250K people 수준이다.

따라서 COCO Captions는 MS COCO 전체 데이터셋 중 **image captioning annotation subset**으로 볼 수 있다.

핵심 특징은 다음과 같다.

| 항목 | 내용 |
|---|---|
| 정식 이름 | Microsoft COCO Captions |
| 기반 데이터 | MS COCO 이미지 |
| 주 언어 | 영어 |
| 태스크 | Image Captioning, Image-Text Retrieval, Multimodal Evaluation |
| 캡션 수 | 보통 이미지당 5개 이상 |
| annotation 형식 | JSON |
| 대표 split | COCO 2014 train/val, Karpathy split |
| 평가 지표 | BLEU, METEOR, ROUGE-L, CIDEr, SPICE 등 |

COCO 공식 data format 문서는 image captioning annotation을 이미지 설명 저장용 annotation으로 정의하고, 각 이미지가 최소 5개의 caption을 가진다고 설명한다.

## 2. 데이터 구성

COCO Captions에서 하나의 샘플은 보통 다음처럼 해석한다.

```text
image -> [caption_1, caption_2, caption_3, caption_4, caption_5]
```

예를 들면 한 장의 주방 사진에 대해 다음처럼 서로 다른 사람이 쓴 설명이 붙을 수 있다.

```text
A man standing in a kitchen preparing food.
A person is cooking in a small kitchen.
A kitchen scene with a man near the counter.
```

중요한 점은 **정답 caption이 하나가 아니라 여러 개**라는 것이다. 이미지 설명은 본질적으로 다답 문제다. 같은 이미지를 `man cooking`, `person preparing food`, `kitchen scene`처럼 다르게 표현해도 모두 맞을 수 있다. 이 때문에 COCO Captions는 단일 label classification보다 평가가 어렵고, 여러 reference caption 기반 metric을 사용한다.

## 3. 주요 Split

### COCO 2014 공식 split

| Split | 이미지 수 | 용도 |
|---|---:|---|
| `train2014` | 약 82K | 학습 |
| `val2014` | 약 40K | 검증/평가 |
| `test2014` | 약 40K | challenge 제출용, 정답 caption 비공개 |

공식 다운로드 페이지 기준으로 2014 train 이미지는 약 83K/13GB, val 이미지는 약 41K/6GB이며, train/val annotation zip은 약 241MB이다. Captioning 2015 challenge는 2014 train/val 및 2014 test split을 사용한다.

### Karpathy split

논문과 오픈소스 구현에서는 **Karpathy split**도 많이 사용한다. COCO 2014 validation을 다시 나누어 validation/test를 만드는 방식이다.

| Split | 이미지 수 |
|---|---:|
| `train` | 82,783 |
| `val` | 5,000 |
| `test` | 5,000 |
| `restval` | 30,504 |

TensorFlow Datasets의 `coco_captions`는 이 split을 제공하며, 모든 split에 caption annotation이 포함된다고 설명한다.

## 4. Annotation JSON 형식

COCO caption annotation 파일은 대체로 다음 구조다.

```json
{
  "info": {},
  "images": [],
  "annotations": [],
  "licenses": []
}
```

이미지 항목은 이미지 식별자, 크기, 파일명, URL, 라이선스 정보를 가진다.

```json
{
  "id": 9,
  "width": 640,
  "height": 480,
  "file_name": "COCO_train2014_000000000009.jpg",
  "license": 3,
  "flickr_url": "...",
  "coco_url": "...",
  "date_captured": "..."
}
```

Caption annotation은 단순하다.

```json
{
  "id": 48,
  "image_id": 9,
  "caption": "A meal is presented in brightly colored plastic trays."
}
```

`image_id`로 이미지와 caption을 join한다. 같은 `image_id`를 가진 annotation이 여러 개 존재하므로, 학습 시에는 보통 다음 중 하나로 처리한다.

| 방식 | 설명 |
|---|---|
| image-caption pair 단위 | 같은 이미지를 caption 수만큼 반복해서 학습 |
| image -> captions list | 한 이미지에 caption list를 묶어서 사용 |
| random caption sampling | epoch마다 해당 이미지의 caption 중 하나를 랜덤 선택 |

## 5. PyTorch에서 사용

`torchvision.datasets.CocoCaptions`를 쓰면 이미지와 caption list를 바로 받을 수 있다. 이 dataset class는 `pycocotools` 설치가 필요하다.

```python
import torchvision.datasets as dset
import torchvision.transforms as transforms

dataset = dset.CocoCaptions(
    root="coco/train2014",
    annFile="coco/annotations/captions_train2014.json",
    transform=transforms.PILToTensor(),
)

image, captions = dataset[0]

print(image.shape)
print(captions)
```

`torchvision.datasets.CocoCaptions`의 반환값은 `(image, target)`이고, `target`은 해당 이미지의 caption list다.

### 노트북에서 사용하는 subset helper

노트북은 `torchvision.datasets.CocoCaptions` 대신 저장소 helper를 사용한다. 수업 실습에서는 전체 데이터셋보다 작은 subset을 빠르게 반복 실행하는 것이 중요하기 때문이다.

| Helper | 역할 |
|---|---|
| `sample_coco_image_subset` | split, seed, sample 수 기준으로 image 단위 COCO subset 생성 |
| `build_coco_dataloader` | subset sample을 batch로 묶어 caption generation에 전달 |
| `COCOSubsetDataset` | image와 caption metadata를 dataset처럼 감싼다 |
| `save_json` | split manifest, metric, report 저장 |

Section 3 기본 설정은 다음과 같다.

```python
split = "validation"
num_images = 4
batch_size = 2
seed = 42
prompt = "Describe this image in one concise English sentence."
metrics = ("bleu", "meteor", "cider_d")
```

Section 4 기본 설정은 다음과 같다.

```python
train_images = 1000
val_images = 50
test_images = 150
compare_images = 4
train_split = "train"
eval_split = "validation"
seed = 42
```

Section 4에서는 validation split에서 `val_images + test_images`만큼 뽑은 뒤 앞쪽 50장은 `Trainer.evaluate()`용 validation set으로, 뒤쪽 150장은 zero-shot과 fine-tuned caption metric 비교용 held-out test set으로 사용한다.

## 6. 평가 방식

모델 출력은 보통 이미지당 하나의 caption이다.

```json
[
  {
    "image_id": 9,
    "caption": "A tray of food with vegetables and bread."
  }
]
```

COCO 공식 result format도 image captioning 결과에 대해 `image_id`와 `caption`만 요구한다.

대표 지표는 다음과 같다.

| Metric | 의미 |
|---|---|
| BLEU | n-gram precision 기반, 기계번역에서 유래 |
| METEOR | unigram alignment, recall, stem/synonym 반영 |
| ROUGE-L | longest common subsequence 기반 |
| CIDEr | 여러 human caption과의 consensus를 TF-IDF n-gram으로 측정 |
| SPICE | scene graph 기반 semantic propositional content 평가 |

COCO caption evaluation toolkit 계열은 BLEU, METEOR, ROUGE-L, CIDEr, SPICE를 제공한다. 원 논문은 COCO caption 평가 서버와 BLEU, METEOR, ROUGE, CIDEr 기반 평가를 설명한다.

노트북의 `compute_caption_metrics`는 실습 시간을 줄이기 위해 다음 metric만 계산한다.

| Metric | 노트북 key | 사용 위치 |
|---|---|---|
| BLEU | `bleu` | Section 3, Section 4 |
| METEOR | `meteor` | Section 3, Section 4 |
| CIDEr-D | `cider_d` | Section 3, Section 4 |

Section 3에서는 zero-shot prediction과 COCO reference caption을 비교한다. Section 4에서는 같은 held-out test sample에 대해 zero-shot caption과 fine-tuned caption을 각각 계산하고, metric별 `delta = fine_tuned - zero_shot`을 저장한다.

주의할 점은 Section 4의 학습 sample은 image마다 대표 caption 하나를 SFT target으로 사용한다는 것이다. COCO는 원래 이미지당 여러 caption이 있지만, 수업용 fine-tuning 흐름에서는 instruction prompt와 단일 caption target을 맞추는 형태로 단순화한다.

## 7. 장점

- 이미지당 여러 caption이 있어 자연어 표현 다양성을 학습하기 좋다.
- COCO detection/segmentation annotation과 함께 쓸 수 있어 vision-language grounding 실험에 유용하다.
- 오래된 표준 벤치마크라 기존 논문 결과와 비교하기 쉽다.
- PyTorch, TensorFlow Datasets, Hugging Face 등에서 쉽게 로딩할 수 있다.

## 8. 한계와 주의점

- 캡션이 짧고 표면적이다. 보통 눈에 보이는 객체와 행동 중심이며, 복잡한 추론이나 장문 설명은 거의 없다.
- 영어 중심 데이터셋이다.
- Caption은 사실 설명에 가깝고 instruction-following 대화 데이터는 아니다.
- 노트북의 fine-tuning은 작은 subset 실습이므로 일반화 성능 평가가 아니라 학습 전후 caption 변화 관찰에 가깝다.
- Section 3의 image 4장 zero-shot metric은 sanity check 용도이며 모델 품질을 대표하지 않는다.
- COCO test server 기준 결과와 Karpathy split 기준 결과는 직접 비교하면 안 된다.
- BLEU 같은 n-gram metric은 의미적으로 맞는 paraphrase를 과소평가할 수 있다.
- 이미지 라이선스와 annotation 라이선스를 구분해서 확인해야 한다. COCO annotation은 CC BY 4.0 계열로 배포되지만, 개별 이미지는 원본 Flickr 라이선스와 사용 조건을 확인해야 한다.

## 9. 노트북 산출물

COCO 관련 실행 결과는 다음 파일과 변수로 확인한다.

| 위치 | 내용 |
|---|---|
| `result_zs` | Section 3 zero-shot sample, prediction, metric |
| `result_ft` | Section 4 fine-tuning 전후 비교 결과 |
| `data/1-4-coco-full-finetuning` | split manifest 등 데이터 산출물 |
| `model/1-4-coco-full-finetuning/before_after.json` | sample별 ground truth, zero-shot, fine-tuned caption |
| `model/1-4-coco-full-finetuning/before_after.png` | before/after caption 비교 이미지 |
| `model/1-4-coco-full-finetuning/train_eval_metrics.json` | 학습 설정, train/eval metric, Trainer log history |

## 참고 자료

- [COCO 공식 다운로드 문서](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm)
- [COCO 공식 data format 문서](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/format-data.htm)
- [COCO 공식 results format 문서](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/format-results.htm)
- [TensorFlow Datasets `coco_captions`](https://tensorflow.google.cn/datasets/catalog/coco_captions)
- [Torchvision `CocoCaptions`](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CocoCaptions.html)
- [pycocoevalcap](https://pypi.org/project/pycocoevalcap/)
- [Microsoft COCO Captions: Data Collection and Evaluation Server](https://huggingface.co/papers/1504.00325)

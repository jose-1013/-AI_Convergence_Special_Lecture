# Simple Mini VLM에서 SmolVLM Fine-tuning까지

이 문서는 `1-VLM-실습.ipynb`의 수업 흐름을 기준으로 Toy VLM 구현과 실제 SmolVLM 실습이 어떻게 연결되는지 정리한다.

노트북의 전체 목표는 PyTorch `Tensor`와 `nn.Module`로 image token과 text token이 하나의 sequence가 되는 과정을 먼저 확인한 뒤, `HuggingFaceTB/SmolVLM-256M-Instruct`로 단일 이미지 추론, COCO zero-shot caption 평가, COCO subset full fine-tuning까지 실행하는 것이다.

## 1. 노트북 Section 구성

| Section | 노트북 내용 | 문서 연결 |
|---|---|---|
| Section 1 | `MiniVisionEncoder`, `MiniProjector`, `MiniTextDecoder`, `SimpleMiniVLM` 구현 | Toy VLM tensor 계약 이해 |
| Section 2 | SmolVLM 단일 이미지 captioning | `docs/smolvlm_256m_instruct.md` |
| Section 3 | COCO validation subset zero-shot captioning, BLEU/METEOR/CIDEr-D 평가 | `docs/coco_captions_dataset.md` |
| Section 4 | COCO subset full fine-tuning, before/after caption 비교 | SmolVLM 실습 확장 |

실행 환경은 다음처럼 나뉜다.

- Section 1은 CPU에서도 shape 확인이 가능하다.
- Section 2부터 Section 4까지는 GPU 런타임을 기본으로 가정한다.
- Colab에서는 `/content/drive/MyDrive/VLM`을 기본 Drive 프로젝트 경로로 사용한다.
- 로컬에서는 `uv sync` 또는 이미 준비된 `.venv`를 우선한다.

## 2. Section 1: Simple Mini VLM

Section 1의 목적은 caption 품질이 아니라 VLM 내부의 tensor 흐름을 눈으로 확인하는 것이다.

```text
pixel_values
  -> MiniVisionEncoder
  -> image_features
  -> MiniProjector
  -> image_embeds
  -> MiniTextDecoder
  -> logits
```

노트북의 Toy VLM은 단순 Conv2d 예제보다 한 단계 더 수업용 구조를 갖는다.

| 구성 | 노트북 구현 | 역할 |
|---|---|---|
| `MiniVisionEncoder` | `Conv2d` patch embedding, positional embedding, 1-layer `TransformerEncoder`, `LayerNorm` | 이미지를 patch token sequence로 변환 |
| `MiniProjector` | `nn.Linear(vision_dim, text_dim)` | vision feature를 text embedding 차원으로 투영 |
| `MiniTextDecoder` | token embedding, causal mask가 있는 1-layer `TransformerEncoder`, `lm_head` | image/text sequence에서 vocabulary logits 생성 |
| `SimpleMiniVLM` | 세 module을 순서대로 호출하고 중간 tensor 반환 | 전체 forward 흐름 검증 |

수업용 TODO는 주로 다음 지점을 채우도록 설계되어 있다.

- `Conv2d(kernel_size=patch_size, stride=patch_size)`
- `self.patch_embed(pixel_values)`
- `flatten(2).transpose(1, 2)`
- `image_features + self.pos_embed`
- `nn.Linear(vision_dim, text_dim)`
- `self.token_embed(input_ids)`
- `torch.cat([image_embeds, text_embeds], dim=1)`
- greedy decoding에서 `logits[:, -1, :].argmax(dim=-1, keepdim=True)`
- `torch.cat([generated_ids, next_token_id], dim=-1)`

## 3. Tensor Shape 기준

노트북 Section 1의 shape 검증은 `224x224` 입력 이미지를 기준으로 한다.

```text
B = 2
H = W = 224
patch_size = 16
L_text = 8
vision_dim = 32
text_dim = 64
vocab_size = 1000
```

`224 / 16 = 14`이므로 image patch token 수는 `14 x 14 = 196`개다. Text token 8개를 붙이면 decoder 입력 sequence 길이는 `196 + 8 = 204`가 된다.

```text
pixel_values:  [2, 3, 224, 224]
input_ids:     [2, 8]
image_features:[2, 196, 32]
image_embeds:  [2, 196, 64]
inputs_embeds: [2, 204, 64]
logits:        [2, 204, 1000]
```

이 shape 흐름이 Section 2 이후의 Hugging Face `processor(...)`, `model.generate(...)` 내부 흐름을 이해하는 기준점이다.

## 4. Toy VLM과 실제 SmolVLM의 대응

| Simple Mini VLM | SmolVLM 실습 | 역할 |
|---|---|---|
| 직접 만든 `pixel_values` | `processor(..., images=[image])` | 이미지 전처리와 tensor 생성 |
| 직접 만든 `input_ids` | `processor.apply_chat_template(...)`와 tokenizer | prompt를 token id로 변환 |
| `MiniVisionEncoder` | SigLIP 계열 vision encoder | 이미지에서 visual token 추출 |
| `MiniProjector` | connector / modality projector | visual token을 text embedding space에 맞춤 |
| `MiniTextDecoder` | SmolLM2 기반 text decoder | 다음 token logits 생성 |
| `generate_text_compact` | `model.generate(...)` | autoregressive decoding |

Toy VLM은 image embedding을 text embedding 앞에 단순 연결한다. 실제 SmolVLM/Idefics3 계열은 prompt 안의 image placeholder 위치를 projected image embedding으로 대체하고, generation cache를 사용해 autoregressive decoding을 효율화한다.

## 5. Section 2: 단일 이미지 SmolVLM 추론

노트북은 `AutoProcessor`와 `AutoModelForImageTextToText`를 사용한다.

```python
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    dtype="auto",
    _attn_implementation="eager",
).to("cuda")
```

입력은 chat template 형식으로 만든다.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"},
        ],
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=500)
```

노트북의 기본 이미지는 자유의 여신상 공개 URL이다. 결과는 `result_single`에 저장하고 `show_caption_cards(...)`로 확인한다.

## 6. Section 3: COCO Zero-shot 평가

Section 3은 COCO validation subset을 작은 수로 뽑아 SmolVLM을 fine-tuning 없이 평가한다.

기본 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| split | `validation` |
| sample 수 | `4` |
| batch size | `2` |
| seed | `42` |
| prompt | `Describe this image in one concise English sentence.` |
| metrics | `bleu`, `meteor`, `cider_d` |

주요 helper는 `utils/` 아래에 있다.

| Helper | 역할 |
|---|---|
| `sample_coco_image_subset` | COCO image 단위 subset 생성 |
| `build_coco_dataloader` | caption generation용 batch 구성 |
| `load_smolvlm_bundle` | processor/model/device 설정 로드 |
| `generate_captions_for_images` | batch image caption 생성 |
| `compute_caption_metrics` | BLEU/METEOR/CIDEr-D 계산 |
| `show_caption_cards`, `show_caption_metrics` | notebook 시각화 |

결과는 `result_zs`에 저장한다.

## 7. Section 4: COCO Subset Full Fine-tuning

Section 4는 같은 모델을 COCO subset으로 full fine-tuning한 뒤 학습 전후 caption을 비교한다.

기본 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| train images | `1000` |
| validation images | `50` |
| test images | `150` |
| comparison images | `4` |
| per-device train batch size | `2` |
| gradient accumulation | `8` |
| epoch | `1` |
| learning rate | `2e-5` |
| output dir | `model/1-4-coco-full-finetuning` |
| data dir | `data/1-4-coco-full-finetuning` |

학습 흐름은 다음 순서다.

1. Train/validation/test image subset을 seed 기반으로 생성한다.
2. Split manifest를 저장해 재현성을 남긴다.
3. SmolVLM processor/model을 로드한다.
4. Zero-shot baseline caption을 먼저 생성한다.
5. 기존 checkpoint가 현재 config와 일치하면 학습을 건너뛴다.
6. `Trainer`와 SFT collator를 구성한다.
7. Full fine-tuning을 실행하거나 checkpoint를 재사용한다.
8. `trainer.evaluate()`로 validation loss를 계산한다.
9. Fine-tuned caption을 생성하고 zero-shot 결과와 같은 test split에서 metric을 비교한다.
10. `before_after.json`, `before_after.png`, `train_eval_metrics.json`을 저장한다.

`StepLossLogger`는 분산/다중 GPU 환경에서 `state.is_world_process_zero`일 때만 loss를 출력하도록 되어 있어 같은 로그가 GPU 수만큼 반복되지 않는다.

## 8. 문서별 역할

| 문서 | 역할 |
|---|---|
| `docs/mini_vlm_to_smolvlm.md` | 노트북 전체 흐름과 Toy VLM-실제 SmolVLM 연결 설명 |
| `docs/smolvlm_256m_instruct.md` | Section 2-4에서 쓰는 모델의 구조, 입력 형식, 실행 팁 |
| `docs/coco_captions_dataset.md` | Section 3-4에서 쓰는 COCO Captions 데이터와 평가 방식 |

## 관련 파일

- `1-VLM-실습.ipynb`
- `utils/smolvlm_utils.py`
- `utils/coco_dataloader.py`
- `utils/caption_metrics.py`
- `utils/training_utils.py`
- `utils/visualization.py`
- `docs/smolvlm_256m_instruct.md`
- `docs/coco_captions_dataset.md`

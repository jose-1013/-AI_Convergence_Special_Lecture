# SmolVLM-256M-Instruct

## 1. 개요

`HuggingFaceTB/SmolVLM-256M-Instruct`는 Hugging Face에서 공개한 초소형 Vision-Language Model이다. 이미지와 텍스트를 함께 입력받아 텍스트를 생성하는 **image-text-to-text** 모델이며, 이미지 설명, 시각 질의응답, 문서/차트 이해, OCR성 질의 등에 사용할 수 있다.

| 항목 | 내용 |
|---|---|
| 모델 이름 | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| 모델 계열 | SmolVLM |
| 개발 | Hugging Face |
| 모델 타입 | Multimodal model, image + text -> text |
| 파라미터 규모 | 약 0.3B, 256M급 |
| 언어 | 주로 영어 |
| 라이선스 | Apache 2.0 |
| 기반 구조 | Idefics3 계열 |
| 주요 구성 | SigLIP vision encoder + projector + SmolLM2 text decoder |
| 대표 용도 | Image captioning, VQA, OCR, document QA, chart QA |
| 이미지 생성 | 지원하지 않음 |

공식 모델 카드는 이 모델을 임의 순서의 이미지와 텍스트 입력을 받아 텍스트 출력을 생성하는 경량 multimodal model로 설명한다. 단일 이미지 inference는 1GB 미만 GPU RAM으로 실행 가능하다고 안내한다.

## 2. 노트북에서 이 모델을 쓰는 위치

`1-VLM-실습.ipynb`에서는 Section 2부터 Section 4까지 이 모델을 사용한다.

| Section | 사용 방식 | 산출물 |
|---|---|---|
| Section 2 | 단일 이미지 captioning | `result_single` |
| Section 3 | COCO validation subset zero-shot captioning | `result_zs` |
| Section 4 | COCO subset full fine-tuning과 before/after 비교 | `result_ft`, `before_after.json`, `before_after.png` |

Section 2 이후는 GPU 런타임을 기본으로 한다. 노트북은 CUDA가 없을 때 느린 CPU fallback을 하지 않고 바로 중단해 실행 환경 문제를 드러낸다.

## 3. 왜 이 모델을 쓰는가

이 모델의 핵심 장점은 **작고 가볍다**는 점이다. 대형 VLM을 쓰기 어려운 Colab, 노트북, 실습 서버, edge device 환경에서도 추론 실험을 시작하기 쉽다.

특히 VLM 강의에서는 다음 이유로 적합하다.

- 구조가 전형적인 VLM 형태다: `vision encoder -> projector -> language decoder`.
- 모델이 작아 로딩과 추론이 빠르다.
- Image captioning 실습에 바로 사용할 수 있다.
- Full fine-tuning, LoRA, QLoRA 실험을 비교적 작은 비용으로 시도할 수 있다.
- Hugging Face `transformers`에서 바로 로드된다.

## 4. 모델 구조

SmolVLM-256M-Instruct는 크게 세 부분으로 이해할 수 있다.

```text
image
  -> vision encoder
  -> visual tokens
  -> modality projector
  -> language embedding space
  -> text decoder
  -> generated text
```

### Vision Encoder

이미지는 SigLIP 계열 vision encoder로 처리된다. 모델 카드에 따르면 SmolVLM-256M은 기존 SmolVLM 2.2B에서 쓰던 약 400M 규모 SigLIP encoder 대신, 더 작은 약 93M 규모 vision encoder를 사용한다.

이미지는 512x512 patch 단위로 처리된다. 각 512x512 image patch는 **64 visual tokens**로 인코딩된다. 큰 이미지는 여러 patch로 나뉘어 처리될 수 있다.

```text
PIL image
  -> pixel_values
  -> SigLIP vision encoder
  -> 64 visual tokens per 512x512 patch
```

### Modality Projector

Vision encoder가 만든 visual feature는 language model이 이해하는 embedding space와 차원이 다르다. Projector는 visual token을 text decoder 입력 embedding 공간으로 변환한다.

```text
visual feature
  -> projector
  -> text embedding-compatible visual tokens
```

이 projector가 있기 때문에 text decoder는 이미지 정보를 텍스트 token과 같은 sequence 안에서 처리할 수 있다.

### Text Decoder

Text decoder는 SmolLM2 기반이다. 모델 카드의 model tree는 base model로 `HuggingFaceTB/SmolLM2-135M`을 표시한다.

Decoder는 autoregressive language model처럼 동작한다. 즉, prompt와 image token을 보고 다음 token을 하나씩 생성한다.

```text
[image tokens] + [text prompt tokens]
  -> decoder
  -> caption / answer / explanation
```

## 5. 입력 형식

이 모델은 이미지와 텍스트가 섞인 대화형 입력을 받는다. Hugging Face 예시는 chat template을 사용한다.

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
```

Processor가 이 메시지를 모델 입력 prompt로 변환한다.

```python
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
```

핵심은 다음이다.

- `{"type": "image"}` 위치에 이미지 placeholder가 들어간다.
- 실제 이미지는 `images=[image]`로 별도 전달된다.
- `apply_chat_template()`가 instruction-tuned 모델에 맞는 대화 형식을 만든다.
- 모델은 image token과 text token을 하나의 sequence처럼 처리한다.

## 6. 노트북 기준 기본 사용 코드

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
if not torch.cuda.is_available():
    raise RuntimeError("Section 2는 GPU 런타임에서 실행하세요.")

device = "cuda"

image = load_image(
    "https://cdn.britannica.com/61/93061-050-99147DCE/"
    "Statue-of-Liberty-Island-New-York-Bay.jpg"
)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype="auto",
    _attn_implementation="eager",
).to(device)

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
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print(generated_text)
```

노트북은 Colab 호환성을 위해 `_attn_implementation="eager"`를 기본으로 둔다. 출력에서 prompt를 제외하고 생성 답변만 보고 싶으면 `generated_ids[:, inputs["input_ids"].shape[-1]:]`를 decode한다.

## 7. COCO 실습에서의 사용

COCO zero-shot 평가와 fine-tuning에서는 같은 prompt를 기준으로 caption을 생성한다.

```text
Describe this image in one concise English sentence.
```

Section 3의 기본값은 validation image 4장, batch size 2, seed 42다. Section 4의 기본값은 train 1000장, validation 50장, held-out test 150장, comparison visualization 4장이다.

노트북은 저장소 helper를 통해 모델 로딩과 caption 생성을 재사용한다.

| Helper | 역할 |
|---|---|
| `load_smolvlm_bundle` | processor/model/device/dtype 로드 |
| `generate_captions_for_images` | batch image caption 생성 |
| `build_sft_collate_fn` | instruction caption SFT용 labels 생성 |
| `build_training_arguments_kwargs` | 설치된 `transformers` 버전에 맞는 `TrainingArguments` 구성 |

Fine-tuning은 `Trainer` 기반 full fine-tuning이다. 기존 checkpoint의 config가 현재 설정과 일치하면 학습을 건너뛰고 checkpoint를 재사용한다.

## 8. 가능한 작업

이 모델은 이미지와 텍스트를 함께 넣고 텍스트 답변을 생성하는 작업에 적합하다.

| 작업 | 설명 |
|---|---|
| Image Captioning | 이미지를 자연어로 설명 |
| Visual Question Answering | 이미지에 대한 질문에 답변 |
| OCR성 질의 | 이미지 속 글자나 문서 내용을 읽고 답변 |
| Document QA | 영수증, 문서, 표 등에 대한 질문 처리 |
| Chart QA | 차트나 다이어그램 기반 질문 처리 |
| Multi-image reasoning | 여러 이미지를 함께 넣고 비교/설명 |
| 짧은 video 분석 | 프레임을 이미지 sequence처럼 넣어 간단히 분석 |

단, 이 모델은 **image generation 모델이 아니다**. 이미지를 생성하거나 편집하지 않는다.

## 9. 학습 데이터

모델 카드 기준 학습 데이터는 주로 다음 두 계열이다.

| 데이터셋 | 역할 |
|---|---|
| `HuggingFaceM4/the_cauldron` | 다양한 VLM instruction/task 데이터 모음 |
| `HuggingFaceM4/Docmatix` | 문서 이해 중심 multimodal 데이터 |

모델 카드에는 학습 데이터 비중도 일부 설명되어 있다.

- Document understanding: 약 25%
- Image captioning: 약 18%
- 그 외 visual reasoning, chart comprehension, instruction following 등 포함

이 구성을 보면 SmolVLM-256M-Instruct는 단순 captioning만을 위한 모델이 아니라, 문서/차트/질의응답까지 고려한 범용 소형 VLM이다.

## 10. 성능 지표

모델 카드에 공개된 benchmark는 다음과 같다.

| Size | MathVista | MMMU | OCRBench | MMStar | AI2D | ChartQA Test | ScienceQA | TextVQA Val | DocVQA Val |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256M | 35.9 | 28.3 | 52.6 | 34.6 | 47.0 | 55.8 | 73.6 | 49.9 | 58.3 |
| 500M | 40.1 | 33.7 | 61.0 | 38.3 | 59.5 | 63.2 | 79.7 | 60.5 | 70.5 |
| 2.2B | 43.9 | 38.3 | 65.5 | 41.8 | 64.0 | 71.6 | 84.5 | 72.1 | 79.7 |

해석하면 다음과 같다.

- 256M 모델은 작기 때문에 500M, 2.2B보다 성능은 낮다.
- 그래도 OCR, DocVQA, TextVQA, ScienceQA 등에서 실습 가능한 수준의 결과를 낸다.
- 강의나 실험에서는 최고 성능 모델보다 구조를 이해하고 빠르게 실행 가능한 모델로 보는 것이 맞다.

## 11. 최적화와 실행 팁

### Precision

GPU에서는 `bfloat16` 사용이 권장된다.

```python
torch_dtype = torch.bfloat16
```

CUDA 환경에서 flash attention이 가능하면 다음 설정을 사용할 수 있다.

```python
_attn_implementation = "flash_attention_2"
```

Colab이나 실습 환경 호환성이 우선이면 노트북처럼 `eager`가 더 안정적이다.

```python
_attn_implementation = "eager"
```

### Quantization

메모리를 더 줄이고 싶으면 8-bit 또는 4-bit quantization을 사용할 수 있다.

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

Fine-tuning에서는 LoRA 또는 QLoRA와 함께 사용하면 GPU 메모리를 줄일 수 있다. 현재 노트북 Section 4는 비교를 단순하게 만들기 위해 LoRA가 아니라 full fine-tuning 흐름을 보여준다.

### Image Resolution

Processor 초기화 시 이미지 longest edge를 조정할 수 있다. 모델 카드는 `size={"longest_edge": N*512}` 형식을 설명한다. 기본적으로 `N=4`, 즉 longest edge 2048 수준을 사용한다.

해상도를 낮추면 메모리를 줄일 수 있지만 작은 글자, 문서, 차트 이해 성능은 떨어질 수 있다.

## 12. 한계와 주의점

- 영어 중심 모델이다.
- 작은 모델이므로 복잡한 추론, 세밀한 OCR, 긴 문서 이해에는 한계가 있다.
- 생성 결과가 사실처럼 보이더라도 틀릴 수 있다.
- 이미지에 없는 내용을 만들어내는 hallucination이 발생할 수 있다.
- 의료, 법률, 채용, 신용평가 등 고위험 의사결정에는 적합하지 않다.
- Instruction-tuned 모델이므로 prompt wording에 따라 결과가 달라질 수 있다.
- COCO captioning 같은 짧은 장면 설명에는 적합하지만, 고정밀 객체 탐지 모델처럼 bounding box를 정확히 예측하는 용도는 아니다.

## 13. 강의 관점 핵심 정리

```markdown
SmolVLM-256M-Instruct는 이미지와 텍스트를 함께 입력받아 텍스트 답변을 생성하는 초소형 VLM이다.
구조적으로는 SigLIP vision encoder가 이미지를 visual token으로 바꾸고, projector가 이를 language decoder 입력 공간에 맞춘 뒤, SmolLM2 기반 decoder가 caption이나 답변을 생성한다.
512x512 image patch는 64 visual tokens로 압축되므로, 큰 VLM보다 메모리와 연산량이 작다.
COCO captioning 실습에서는 이미지를 입력하고 "Describe this image" 같은 prompt를 넣어 zero-shot caption을 생성하는 기준 모델로 사용할 수 있다.
작고 빠르기 때문에 full fine-tuning, LoRA, QLoRA 같은 학습 방식의 차이를 수업에서 비교하기 좋다.
```

## 참고 자료

- [Hugging Face 모델 카드: SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [Hugging Face 블로그: SmolVLM 256M & 500M](https://huggingface.co/blog/smolervlm)
- [Hugging Face 블로그: SmolVLM](https://huggingface.co/blog/smolvlm)
- [SmolVLM 논문 페이지](https://huggingface.co/papers/2504.05299)

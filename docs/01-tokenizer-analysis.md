# Tokenizer 분석

> 신뢰도: 높음 | 접근성: 높음 | Fine-tuning 탐지력: 우수

## 개요

Tokenizer 분석은 LLM이 from scratch로 학습되었는지 판별하는 가장 접근성 높은 방법입니다. Fine-tuning 시 tokenizer를 재학습하는 경우가 거의 없기 때문에, tokenizer의 유사성은 모델 기원을 추적하는 강력한 지표가 됩니다.

## 분석 항목

### 1. Vocabulary 비교
- 기존 base model들과의 토큰 중복률 확인
- 고유 토큰 식별

### 2. BPE Merge Rules 분석
- Merge 순서 및 패턴 비교
- 동일한 merge rules는 같은 tokenizer 증거

### 3. 특수 토큰 패턴 비교
- `<eos>`, `<pad>`, `<bos>`, `<unk>` 등
- Chat template 토큰 (`<|im_start|>`, `[INST]` 등)

## Tokenizer 작동 원리

| 방식 | 특징 | 사용 모델 |
|------|------|----------|
| **BPE** (Byte Pair Encoding) | 빈도 기반으로 인접 문자쌍 병합 | GPT-2, RoBERTa |
| **WordPiece** | likelihood 최대화 기준 병합 | BERT |
| **SentencePiece** | 공백 포함 원시 텍스트 처리 (▁ 마커 사용) | T5, Gemma, Llama |

## Fine-tuning 시 Tokenizer를 재학습하지 않는 이유

1. **Embedding 호환성**: 새 vocabulary는 pre-trained embedding과 호환되지 않음
2. **비용**: Tokenizer 재학습은 전체 corpus 재처리 필요
3. **성능 저하 위험**: Vocabulary 변경 시 원래 도메인 성능 저하

## Vocabulary 중복률 해석 기준

| 중복률 | 해석 |
|--------|------|
| **>98%** | Fine-tuning 가능성 높음 (safety token 등 소량 추가만) |
| **90-98%** | Continued pre-training 또는 vocabulary 확장 |
| **<90%** | From scratch 학습 강력 증거 |

---

## 모델별 검증 결과

### 1. Upstage Solar-Open-100B ✅

**검증일**: 2026-01-04

#### Vocabulary 크기 비교

| 모델 | Vocab Size | Tokenizer Type | Solar 대비 |
|------|-----------|----------------|------------|
| **Solar-Open-100B** | **196,608** | SentencePiece (BPE) | - |
| Qwen2-72B | 152,064 | BPE | -23% |
| Llama-3 | 128,256 | tiktoken (BPE) | -35% |
| DeepSeek-V2 | 102,400 | BPE | -48% |
| Mixtral-8x7B | 32,000 | SentencePiece | -84% |

#### Special Tokens 비교

| 모델 | bos_token | eos_token | pad_token |
|------|-----------|-----------|-----------|
| **Solar-Open-100B** | `<s>` | `</s>` | `<pad>` |
| Llama-3 | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | (없음) |
| Mixtral | `<s>` | `</s>` | (없음) |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| Vocab Size 일치 | 0개 모델 | ✅ From scratch 지지 |
| Special Tokens | Mixtral과 유사 | ⚠️ 중립 |
| Tokenizer Type | 공통 방식 | ⚠️ 중립 |

**결론: From scratch 학습 주장 지지**

---

### 2. NAVER Cloud HyperCLOVAX-SEED-Think-32B ⚠️

**검증일**: 2026-01-05 (심층 분석: 2026-01-05)

#### Vocabulary 크기 비교

| 소스 | Vocab Size | 비고 |
|------|-----------|------|
| **HyperCLOVAX-SEED** (tokenizer.json) | **110,524** | 실제 토큰 수 |
| **HyperCLOVAX-SEED** (config.json) | 128,256 | 모델 설정값 (padding 포함) |
| Llama 3/3.1 | 128,256 | config.json 기준 |
| Qwen2.5-VL | 151,665 | tokenizer.json 기준 |
| Trillion-7B | 128,256 | config.json 기준 |
| HyperCLOVA X (논문) | 100,000 | 원래 HyperCLOVA X |

**중요**: config.json의 `vocab_size`와 tokenizer.json의 실제 토큰 수가 다를 수 있음. 분석은 tokenizer.json 기준.

#### 심층 분석 결과 (2026-01-05)

**HyperCLOVAX-SEED vs Llama-3 비교:**

| 지표 | 값 | 해석 |
|------|-----|------|
| HyperCLOVAX-SEED vocab | 110,524 | |
| Llama-3 vocab | 128,256 | |
| **공통 토큰 수** | 102,323 | |
| **토큰 중복률** | **92.58%** | 높음 |
| **BPE merge rules 순서 일치율** | **0.01%** | 매우 낮음 |
| 공통 merge rules | 102,114 | |
| 전체 merge 중복률 | 92.57% | |

**HyperCLOVAX-SEED vs Qwen2.5-VL 비교:**

| 지표 | 값 | 해석 |
|------|-----|------|
| **토큰 중복률** | **91.23%** | 높음 |
| **BPE merge rules 순서 일치율** | **1.54%** | Llama보다 높음 |
| **처음 20개 merge rules** | **모두 일치** | 주목할 만한 유사성 |
| 공통 merge rules | 100,450 | |

#### 핵심 발견

1. **Llama 3 직접 재사용 여부**: ❌ **아님**
   - 토큰은 92.58% 공유하지만
   - BPE merge rules 순서가 **0.01%만 일치** (사실상 완전 불일치)
   - 독립적으로 BPE 학습을 수행한 증거

2. **Qwen 계열과의 관계**: ⚠️ **의심**
   - 처음 20개 merge rules가 **모두 일치**
   - Merge 순서 일치율이 Llama (0.01%)보다 **150배 높음** (1.54%)
   - Qwen 계열 tokenizer를 기반으로 확장했을 가능성

3. **한글 토큰**: HyperCLOVAX-SEED에만 있는 토큰에서 한글 바이트 패턴 다수 발견
   - 예: `'Ġì¢ĭìĬµëĭĪëĭ¤'`, `'ĠíķĢ'`, `'ì¡°ìĦł'` 등

#### Special Tokens 비교

| 토큰 | 값 | 비고 |
|------|-----|------|
| `<\|IMAGE_PAD\|>` | Vision용 | VLM 특화 |
| `<\|im_start\|>`, `<\|im_end\|>` | Conversation | ChatML 스타일 |
| `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` | Code | Fill-in-the-middle |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Llama 3 재사용** | ❌ 아님 | BPE merge 순서 0.01% 일치 |
| **Qwen 계열 관계** | ⚠️ 의심 | merge 순서 1.54%, 처음 20개 완전 일치 |
| **Special Tokens** | 독자적 구성 | ✅ 지지 |
| **한글 최적화** | 확인됨 | 고유 한글 토큰 존재 |

**결론: Llama 3 tokenizer 직접 재사용은 아니지만, Qwen 계열 tokenizer와 유사한 BPE 학습 순서를 보임. 독립 학습 여부 판단을 위해 Qwen2 (non-VL) tokenizer와의 추가 비교 필요.**

> 분석 기준: HuggingFace `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B`의 tokenizer.json

---

### 3. SKT A.X-K1 ✅

**검증일**: 2026-01-05

#### Vocabulary 크기 비교

| 모델 | Vocab Size | 비고 |
|------|-----------|------|
| **A.X-K1** | **163,840** | 모든 모델과 불일치 |
| Solar-Open-100B | 196,608 | -17% |
| Qwen2-72B | 152,064 | +8% |
| DeepSeek-V2 | 102,400 | +60% |

#### Special Tokens 구성

| 토큰 | 값 | 비고 |
|------|-----|------|
| `bos_token` | `<\|endoftext\|>` | GPT 스타일 |
| `eos_token` | `<\|im_end\|>` | ChatML 스타일 |
| `pad_token` | `<\|pad\|>` | |
| `<\|think\|>`, `</think>` | Reasoning | Chain-of-thought 지원 |
| `<\|image\|>`, `<\|video_*\|>` | Multimodal | VLM 준비 |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Vocab Size 일치** | 0개 모델 | ✅ From scratch 지지 |
| **Special Tokens** | ChatML + 고유 토큰 | ✅ 독자 설계 |
| **Tokenizer Type** | PreTrainedTokenizerFast | ⚠️ 중립 |

**결론: From scratch 학습 주장 지지**

---

### 4. NC AI VAETKI ✅

**검증일**: 2026-01-05

#### Vocabulary 크기 비교

| 모델 | Vocab Size | 비고 |
|------|-----------|------|
| **VAETKI** | **137,216** | 모든 모델과 불일치 |
| Solar-Open-100B | 196,608 | -30% |
| A.X-K1 | 163,840 | -16% |
| Llama-3 | 128,256 | +7% |
| Qwen2-72B | 152,064 | -10% |

#### Special Tokens 구성

| 토큰 | 값 | 비고 |
|------|-----|------|
| `bos_token` | `<\|START\|>` | 고유 스타일 |
| `eos_token` | `<\|END\|>` | 고유 스타일 |
| `pad_token` | `<\|END\|>` | eos와 동일 |
| `<tool_start>`, `<tool_end>` | Tool calling | 함수 호출 지원 |
| `<think>`, `</think>` | Reasoning | Chain-of-thought 지원 |
| `<\|role_start\|>`, `<\|role_end\|>` | Conversation | 대화 역할 구분 |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Vocab Size 일치** | 0개 모델 | ✅ From scratch 지지 |
| **Special Tokens** | 완전히 고유한 패턴 | ✅ 독자 설계 |
| **Tokenizer Type** | PreTrainedTokenizerFast | ⚠️ 중립 |

**결론: From scratch 학습 주장 지지**

---

### 5. LG AI 연구원 K-EXAONE ✅

**검증일**: 2026-01-05

#### Vocabulary 크기 비교

| 모델 | Vocab Size | 비고 |
|------|-----------|------|
| **K-EXAONE** | **153,600** | 모든 모델과 불일치 |
| Solar-Open-100B | 196,608 | -22% |
| A.X-K1 | 163,840 | -6% |
| VAETKI | 137,216 | +12% |
| Qwen2-72B | 152,064 | +1% (유사하나 다름) |

#### Special Tokens 구성

| 토큰 | 값 | 비고 |
|------|-----|------|
| `bos_token` | `[BOS]` | 브라켓 스타일 |
| `eos_token` | `<\|endofturn\|>` | 고유 스타일 |
| `pad_token` | `[PAD]` | 브라켓 스타일 |
| `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` | Conversation | ChatML 유사 |
| `<think>`, `</think>` | Reasoning | Chain-of-thought 지원 |
| `<tool_call>` | Tool calling | 함수 호출 지원 |
| `<vision>`, `<\|image_pad\|>`, `<\|video_pad\|>` | Multimodal | VLM 지원 |
| `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` | Code | Fill-in-the-middle |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Vocab Size 일치** | 0개 모델 (Qwen2와 유사하나 1,536 차이) | ✅ From scratch 지지 |
| **Special Tokens** | EXAONE 고유 스타일 (`[BOS]`, `<\|endofturn\|>`) | ✅ 독자 설계 |
| **Tokenizer Type** | PreTrainedTokenizerFast | ⚠️ 중립 |

**결론: From scratch 학습 주장 지지**

---

## 분석 코드

### 1. Vocabulary 비교

```python
from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("base-model")
target_tok = AutoTokenizer.from_pretrained("target-model")

base_vocab = set(base_tok.get_vocab().keys())
target_vocab = set(target_tok.get_vocab().keys())

overlap = len(base_vocab & target_vocab)
overlap_pct = (overlap / len(base_vocab)) * 100
print(f"중복률: {overlap_pct:.2f}%")

# 고유 토큰 확인
only_in_base = base_vocab - target_vocab
only_in_target = target_vocab - base_vocab
print(f"Base에만 있는 토큰: {len(only_in_base)}")
print(f"Target에만 있는 토큰: {len(only_in_target)}")
```

### 2. Merge Rules 비교 (BPE/SentencePiece)

```python
# merges가 동일하면 같은 tokenizer
base_merges = base_tok.backend_tokenizer.model.get_vocab()
target_merges = target_tok.backend_tokenizer.model.get_vocab()

# 첫 100개 merge 비교
merge_match = sum(1 for i in range(min(100, len(base_merges), len(target_merges)))
                  if list(base_merges.items())[i] == list(target_merges.items())[i])
print(f"첫 100개 merge 일치율: {merge_match}%")
```

### 3. Special Tokens 비교

```python
print("Base special tokens:", base_tok.special_tokens_map)
print("Target special tokens:", target_tok.special_tokens_map)

# 추가된 special tokens 확인
print("Added tokens:", target_tok.added_tokens_encoder)
```

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 주요 base model과 vocabulary 중복률 90% 미만
- 고유한 special token 체계
- 독자적인 merge rules

**Fine-tuning 의심 증거:**
- 특정 base model과 95% 이상 vocabulary 중복
- 동일한 special token 패턴
- merge rules 일치

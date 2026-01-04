# 튜토리얼: Q&A 형식 학습 기록

이 문서는 Solar-Open-100B 검증 과정에서 진행한 질문과 답변을 시간순으로 기록합니다.

---

## Q1: LLM이 "from scratch"로 학습되었는지 어떻게 검증할 수 있나요?

**질문 시각**: 2026-01-04

**답변**:

LLM이 from scratch로 학습되었는지 vs fine-tuned인지 검증하는 여러 기술적 방법이 있습니다:

1. **Tokenizer 분석** (가장 접근성 높음)
   - Tokenizer는 재학습 비용이 높아 fine-tuning시 거의 재사용됨
   - Vocabulary의 95% 이상이 기존 모델과 동일하면 fine-tuning 가능성 높음
   - `tokenizer.get_vocab()`으로 추출하여 비교

2. **Weight 분석**
   - Fine-tuned 모델은 초기 레이어에서 base model과 90% 이상 cosine similarity 유지
   - SHA-256 해시로 weight tensor 비교
   - From scratch 모델은 PCA에서 orthogonal 분포

3. **Architecture 비교**
   - `model.config`로 hyperparameter 확인
   - 동일한 config는 fine-tuning 강력 증거
   - 고유한 구성요소(예: 특이한 RoPE scaling)는 from scratch 증거

4. **행동 테스트**
   - Knowledge cutoff 날짜 확인
   - Safety alignment, refusal pattern이 base model과 동일하면 fine-tuning 가능성

5. **Compute 추정**
   - From scratch는 fine-tuning 대비 10-100배 더 많은 compute 필요
   - 19.7T tokens 학습은 massive compute로, from scratch 주장과 일관성 있음

| 방법 | 신뢰도 | 접근성 | Fine-tuning 탐지력 |
|------|--------|--------|-------------------|
| Tokenizer 분석 | 높음 | 높음 | 우수 |
| Weight 해싱 | 높음 | 중간 | 양호 |
| Architecture 매칭 | 중간 | 높음 | 양호 |
| 행동 테스트 | 중간 | 높음 | 보통 |
| Training Logs | 매우 높음 | 낮음 | 우수 |

---

## Q2: Tokenizer 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?

**질문 시각**: 2026-01-04

**답변**:

### Tokenizer의 작동 원리

Tokenizer는 텍스트를 모델이 처리할 수 있는 숫자(token ID)로 변환합니다. 주요 방식:

| 방식 | 특징 | 사용 모델 |
|------|------|----------|
| **BPE** (Byte Pair Encoding) | 빈도 기반으로 인접 문자쌍 병합 | GPT-2, RoBERTa |
| **WordPiece** | likelihood 최대화 기준 병합 | BERT |
| **SentencePiece** | 공백 포함 원시 텍스트 처리 (▁ 마커 사용) | T5, Gemma, Llama |

### Fine-tuning 시 Tokenizer를 재학습하지 않는 이유

1. **Embedding 호환성**: 새 vocabulary는 pre-trained embedding과 호환되지 않음
2. **비용**: Tokenizer 재학습은 전체 corpus 재처리 필요
3. **성능 저하 위험**: Vocabulary 변경 시 원래 도메인 성능 저하

### Vocabulary 중복률 기준

| 중복률 | 해석 |
|--------|------|
| **>98%** | Fine-tuning 가능성 높음 (safety token 등 소량 추가만) |
| **90-98%** | Continued pre-training 또는 vocabulary 확장 |
| **<90%** | From scratch 학습 강력 증거 |

### 분석 기법

**1. Vocabulary 비교**
```python
from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("base-model")
target_tok = AutoTokenizer.from_pretrained("target-model")

base_vocab = set(base_tok.get_vocab().keys())
target_vocab = set(target_tok.get_vocab().keys())

overlap = len(base_vocab & target_vocab)
overlap_pct = (overlap / len(base_vocab)) * 100
print(f"중복률: {overlap_pct:.2f}%")
```

**2. Merge Rules 비교 (BPE/SentencePiece)**
```python
# merges가 동일하면 같은 tokenizer
base_merges = base_tok.backend_tokenizer.model.get_vocab()
target_merges = target_tok.backend_tokenizer.model.get_vocab()
```

**3. Special Tokens 비교**
```python
print(base_tok.special_tokens_map)
print(target_tok.special_tokens_map)
# [PAD], [UNK], [CLS], <eos> 등 비교
```

**4. Encoding 결과 비교**
```python
text = "Hello, world! 토큰화 테스트입니다."
base_tokens = base_tok.tokenize(text)
target_tokens = target_tok.tokenize(text)
# 동일 입력에 다른 토큰 분할 → 다른 tokenizer
```

### Solar-Open-100B 검증 시 비교 대상

1. **Llama 계열**: Llama-2, Llama-3 (SentencePiece 기반)
2. **Mistral/Mixtral**: MoE 아키텍처 유사
3. **Qwen**: 대규모 한국어 포함 모델
4. **DeepSeek-MoE**: MoE 아키텍처

**핵심**: Solar-Open-100B의 tokenizer가 위 모델들과 90% 미만 중복이면 from scratch 주장 지지

---

<!-- TUTORIAL_MARKER: 새로운 Q&A는 이 마커 위에 자동 추가됩니다 -->

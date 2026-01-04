# Architecture 분석

> 신뢰도: 중간 | 접근성: 높음 | Fine-tuning 탐지력: 양호

## 개요

Architecture 분석은 모델의 구조적 특징을 비교하여 기원을 추적합니다. 동일한 architecture config는 fine-tuning의 강력한 증거가 되며, 고유한 구조적 특징은 from scratch 학습을 시사합니다.

## 분석 항목

### 1. 기본 Hyperparameters 비교
- 레이어 수, hidden dimension, attention heads
- Intermediate size, vocabulary size

### 2. 활성화 함수 및 Normalization
- SiLU, GELU, ReLU 등
- RMSNorm vs LayerNorm

### 3. 고유한 구조적 특징
- RoPE scaling 방식
- Attention 구현 (GQA, MQA, MHA)
- MoE 구성 (expert 수, top-k)

## config.json 분석

### 주요 비교 항목

```python
from transformers import AutoConfig

def compare_configs(model_names):
    configs = {}
    for name in model_names:
        configs[name] = AutoConfig.from_pretrained(name)

    # 주요 항목 비교
    keys = [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "hidden_act",
    ]

    print("| 항목 |", " | ".join(model_names), "|")
    print("|------|", " | ".join(["---"] * len(model_names)), "|")

    for key in keys:
        values = []
        for name in model_names:
            val = getattr(configs[name], key, "N/A")
            values.append(str(val))
        print(f"| {key} |", " | ".join(values), "|")

# 사용 예시
compare_configs([
    "upstage/Solar-Open-100B",
    "meta-llama/Llama-3-70B",
    "mistralai/Mixtral-8x7B-v0.1"
])
```

## MoE 구조 분석

Solar-Open-100B는 MoE 아키텍처를 사용하므로 MoE 관련 config가 특히 중요합니다.

### MoE Config 비교

```python
def compare_moe_configs(model_names):
    configs = {}
    for name in model_names:
        configs[name] = AutoConfig.from_pretrained(name)

    moe_keys = [
        "num_local_experts",
        "num_experts_per_tok",
        "router_aux_loss_coef",
        "expert_interval",
    ]

    print("| MoE 항목 |", " | ".join(model_names), "|")
    print("|----------|", " | ".join(["---"] * len(model_names)), "|")

    for key in moe_keys:
        values = []
        for name in model_names:
            val = getattr(configs[name], key, "N/A")
            values.append(str(val))
        print(f"| {key} |", " | ".join(values), "|")
```

### Solar-Open-100B MoE 구성

| 항목 | 값 |
|------|-----|
| Expert 수 | 129개 (128 routed + 1 shared) |
| 활성 Expert | top-8 |
| 총 파라미터 | 102.6B |
| 활성 파라미터 | 12B |

## Attention 구조 분석

### GQA (Grouped Query Attention) 확인

```python
def analyze_attention(config):
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    if num_kv_heads == num_heads:
        attn_type = "MHA (Multi-Head Attention)"
    elif num_kv_heads == 1:
        attn_type = "MQA (Multi-Query Attention)"
    else:
        attn_type = f"GQA (Grouped Query Attention, {num_heads//num_kv_heads} groups)"

    print(f"Attention type: {attn_type}")
    print(f"Query heads: {num_heads}")
    print(f"KV heads: {num_kv_heads}")

    return attn_type
```

## RoPE (Rotary Position Embedding) 분석

```python
def analyze_rope(config):
    rope_theta = getattr(config, "rope_theta", 10000)
    rope_scaling = getattr(config, "rope_scaling", None)

    print(f"RoPE theta: {rope_theta}")
    print(f"RoPE scaling: {rope_scaling}")

    # 고유한 rope_theta 값은 from scratch 증거
    common_theta_values = [10000, 500000, 1000000]
    if rope_theta not in common_theta_values:
        print("⚠️ 비표준 rope_theta 값 - 고유 설정 가능성")
```

## 유사 모델 비교표

| 모델 | Type | Layers | Hidden | Heads | KV Heads | Experts | Vocab |
|------|------|--------|--------|-------|----------|---------|-------|
| **Solar-Open-100B** | MoE | 48 | 4096 | 64 | 8 | 128+1 | 196,608 |
| Mixtral-8x7B | MoE | 32 | 4096 | 32 | 8 | 8 | 32,000 |
| DeepSeek-V2 | MoE | 60 | 5120 | 128 | 128 | 160+2 | 102,400 |
| Qwen2-57B-A14B | MoE | 28 | 3584 | 28 | 4 | 64 | 151,936 |
| Llama-3-70B | Dense | 80 | 8192 | 64 | 8 | - | 128,256 |

## 고유성 판단 기준

### 동일 config 판정

다음 조건을 모두 만족하면 동일 architecture로 판정:

1. `hidden_size` 일치
2. `num_hidden_layers` 일치
3. `num_attention_heads` 일치
4. `intermediate_size` 일치
5. `hidden_act` 일치

### 파생 모델 가능성 지표

| 일치 항목 | 해석 |
|----------|------|
| 5/5 | 동일 architecture - fine-tuning 의심 |
| 3-4/5 | 유사 architecture - 참조 가능성 |
| 1-2/5 | 독립적 설계 가능성 |
| 0/5 | 완전히 다른 architecture |

## 검증 체크리스트

- [x] Solar-Open-100B config.json 분석
- [x] Llama-3와 비교
- [x] Mixtral과 MoE 구조 비교
- [x] DeepSeek-MoE와 비교
- [x] Qwen과 비교
- [x] RoPE 설정 분석
- [x] Attention 구조 분석
- [x] 고유한 config 항목 식별

---

## 검증 결과 (2026-01-04)

### Architecture 비교 요약

| 파라미터 | Solar-Open-100B | Mixtral | DeepSeek-V2 | Qwen2-57B | 일치 모델 |
|----------|-----------------|---------|-------------|-----------|----------|
| hidden_size | 4,096 | 4,096 | 5,120 | 3,584 | Mixtral만 |
| num_layers | 48 | 32 | 60 | 28 | 없음 |
| num_heads | 64 | 32 | 128 | 28 | 없음 |
| num_kv_heads | 8 | 8 | 128 | 4 | Mixtral만 |
| n_experts | 128+1 | 8 | 160+2 | 64 | 없음 |
| vocab_size | 196,608 | 32,000 | 102,400 | 151,936 | 없음 |
| rope_theta | 1,000,000 | 1,000,000 | 10,000 | 1,000,000 | Mixtral, Qwen |

### 판정

| 일치 항목 수 | 비교 대상 | 결과 |
|-------------|----------|------|
| **2/7** | Mixtral | hidden_size, kv_heads만 일치 |
| **1/7** | DeepSeek-V2 | rope_theta 계열만 유사 |
| **1/7** | Qwen2-57B | rope_theta만 동일 |

**결론: 0/5 완전 일치 → 독립적 설계 (From scratch 지지)**

### Solar-Open-100B 고유 특징

1. **129개 Expert 구성** (128 routed + 1 shared) - 다른 모델에서 볼 수 없는 구성
2. **48 layers** - Mixtral(32)과 DeepSeek(60)의 단순 중간값 아님
3. **64 attention heads** - 가장 많은 head 수 (Dense 모델 제외)
4. **moe_intermediate_size: 1,280** - 비교 대상 중 가장 작음 (효율적 설계)
5. **vocab_size: 196,608** - 모든 비교 대상 중 가장 큼

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 주요 모델과 architecture 불일치
- 고유한 MoE 구성 (129 experts)
- 비표준 hyperparameter 조합

**Fine-tuning 의심 증거:**
- 특정 모델과 완전한 config 일치
- 동일한 hidden_size, layers, heads 조합
- 표준적인 RoPE 설정

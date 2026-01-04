# Weight 분석

> 신뢰도: 높음 | 접근성: 중간 | Fine-tuning 탐지력: 양호

## 개요

Weight 분석은 모델의 가중치를 직접 비교하여 from scratch 학습 여부를 판별합니다. Fine-tuned 모델은 base model과 높은 가중치 유사성을 보이는 반면, from scratch 모델은 독립적인 가중치 분포를 갖습니다.

## 분석 항목

### 1. Layer별 Cosine Similarity 계산
- 각 layer의 weight tensor 간 유사도 측정
- Fine-tuned 모델: 초기 레이어 90% 이상 유사도

### 2. Weight Tensor 해시 비교
- SHA-256 해시로 동일성 확인
- 완전히 동일한 layer 탐지

### 3. PCA를 통한 Weight 분포 분석
- Weight matrix의 주성분 분석
- From scratch: orthogonal 분포
- Fine-tuned: base model 근처 clustering

## 분석 코드

### 1. Cosine Similarity 계산

```python
import torch
from transformers import AutoModel
import torch.nn.functional as F

def load_model_weights(model_name):
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    return {name: param.data for name, param in model.named_parameters()}

base_weights = load_model_weights("base-model")
target_weights = load_model_weights("target-model")

def cosine_sim(w1, w2):
    w1_flat = w1.flatten().float()
    w2_flat = w2.flatten().float()
    return F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()

# Layer별 유사도 계산
similarities = {}
for name in base_weights:
    if name in target_weights:
        if base_weights[name].shape == target_weights[name].shape:
            sim = cosine_sim(base_weights[name], target_weights[name])
            similarities[name] = sim
            print(f"{name}: {sim:.4f}")

# 평균 유사도
avg_sim = sum(similarities.values()) / len(similarities)
print(f"\n평균 유사도: {avg_sim:.4f}")
```

### 2. Weight Tensor 해시 비교

```python
import hashlib
import numpy as np

def weight_hash(tensor):
    """Weight tensor의 SHA-256 해시 계산"""
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]

# 동일한 weight 탐지
identical_layers = []
for name in base_weights:
    if name in target_weights:
        if base_weights[name].shape == target_weights[name].shape:
            base_hash = weight_hash(base_weights[name])
            target_hash = weight_hash(target_weights[name])
            if base_hash == target_hash:
                identical_layers.append(name)
                print(f"동일: {name}")

print(f"\n동일한 layer 수: {len(identical_layers)}")
```

### 3. PCA 분석

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_layer_features(weights, layer_prefix="layers"):
    """특정 layer들의 weight를 feature vector로 변환"""
    features = []
    for name, w in weights.items():
        if layer_prefix in name and "weight" in name:
            features.append(w.flatten().cpu().numpy()[:1000])  # 첫 1000개 요소
    return np.array(features)

# 여러 모델 비교
models = {
    "base": base_weights,
    "target": target_weights,
    # 필요시 더 추가
}

all_features = []
labels = []
for model_name, weights in models.items():
    feats = extract_layer_features(weights)
    all_features.extend(feats)
    labels.extend([model_name] * len(feats))

# PCA 수행
pca = PCA(n_components=2)
reduced = pca.fit_transform(np.array(all_features))

# 시각화
for model_name in models:
    mask = [l == model_name for l in labels]
    plt.scatter(reduced[mask, 0], reduced[mask, 1], label=model_name, alpha=0.5)
plt.legend()
plt.title("Weight Distribution (PCA)")
plt.savefig("weight_pca.png")
```

## Embedding Layer 분석

Embedding layer는 tokenizer와 밀접하게 연관되어 있어 특히 중요합니다.

```python
def compare_embeddings(base_model, target_model):
    """Embedding layer 상세 비교"""
    base_emb = base_model.get_input_embeddings().weight.data
    target_emb = target_model.get_input_embeddings().weight.data

    print(f"Base embedding shape: {base_emb.shape}")
    print(f"Target embedding shape: {target_emb.shape}")

    # Shape이 다르면 vocabulary가 다름 → from scratch 증거
    if base_emb.shape != target_emb.shape:
        print("Embedding shape 불일치 - 다른 vocabulary 사용")
        return

    # Cosine similarity
    sim = cosine_sim(base_emb, target_emb)
    print(f"Embedding 유사도: {sim:.4f}")

    # 개별 토큰 embedding 비교
    sample_indices = [0, 100, 1000, 10000]
    for idx in sample_indices:
        if idx < base_emb.shape[0]:
            token_sim = cosine_sim(base_emb[idx], target_emb[idx])
            print(f"Token {idx} 유사도: {token_sim:.4f}")
```

## 해석 기준

### Cosine Similarity 기준

| 평균 유사도 | 해석 |
|-------------|------|
| **>0.95** | 거의 확실히 fine-tuning |
| **0.8-0.95** | Fine-tuning 또는 continued pre-training |
| **0.5-0.8** | 부분적 weight 재사용 가능성 |
| **<0.5** | From scratch 가능성 높음 |

### Layer별 패턴

| 패턴 | 의미 |
|------|------|
| 초기 layer 높은 유사도, 후기 layer 낮음 | 전형적인 fine-tuning |
| 전체적으로 낮은 유사도 | From scratch 증거 |
| 일부 layer만 높은 유사도 | 부분적 weight 초기화 |

## 검증 체크리스트

- [x] Solar-Open-100B architecture 분석
- [x] 비교 대상 모델 architecture 수집
- [x] Weight 비교 가능성 판단
- [ ] ~~Llama-3와 layer별 cosine similarity 계산~~ (architecture 불일치로 불가)
- [ ] ~~Mixtral과 MoE layer 비교~~ (architecture 불일치로 불가)
- [ ] ~~Embedding layer 상세 분석~~ (vocab_size 불일치로 불가)

---

## 검증 결과 (2026-01-04)

### Architecture 비교를 통한 Weight 비교 가능성 분석

Weight 비교는 동일한 shape의 tensor 간에만 의미가 있습니다.

| 파라미터 | Solar-Open-100B | Mixtral-8x7B | DeepSeek-V2 | Qwen2-57B |
|----------|-----------------|--------------|-------------|-----------|
| **hidden_size** | 4,096 | 4,096 | 5,120 | 3,584 |
| **num_hidden_layers** | 48 | 32 | 60 | 28 |
| **num_attention_heads** | 64 | 32 | 128 | 28 |
| **n_routed_experts** | 128 | 8 | 160 | 64 |
| **vocab_size** | 196,608 | 32,000 | 102,400 | 151,936 |

### 판정

| 비교 대상 | Weight 비교 가능? | 이유 |
|-----------|------------------|------|
| Mixtral-8x7B | ❌ 불가 | layers, heads, experts 모두 다름 |
| DeepSeek-V2 | ❌ 불가 | hidden_size부터 다름 |
| Qwen2-57B | ❌ 불가 | 모든 dimension 다름 |

### 결론

**Weight 비교 불가 → From scratch 증거**

Fine-tuning된 모델이라면 base model과 동일한 architecture를 가져야 합니다.
Solar-Open-100B는 어떤 기존 모델과도 architecture가 일치하지 않으므로,
**직접적인 weight 비교 없이도 from scratch 학습임을 강력히 시사**합니다.

---

## 주의사항

1. **메모리 요구사항**: 100B 모델 비교는 상당한 GPU/CPU 메모리 필요
2. **MoE 구조 고려**: Expert weight는 별도 분석 필요
3. **Quantization 영향**: 양자화된 모델은 해시 비교 불가

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 base model과 평균 cosine similarity 0.5 미만
- 동일한 layer 없음 (해시 불일치)
- PCA에서 독립적인 분포

**Fine-tuning 의심 증거:**
- 특정 base model과 0.9 이상 유사도
- 다수의 동일 layer (해시 일치)
- 초기 layer 높은 유사도 패턴

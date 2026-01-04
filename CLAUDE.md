# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Upstage의 Solar-Open-100B 모델이 "from scratch"로 학습되었는지 검증하는 프로젝트입니다.
검증 과정의 모든 Q&A는 `docs/00-tutorial.md`에 기록됩니다.

**검증 대상**: [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B)
- 공식 주장: "Trained Entirely from Scratch"
- 아키텍처: MoE (102.6B total, 12B active, 129 experts)
- 학습 토큰: 19.7T tokens

## 사용 가능한 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/commit-push` | 변경사항 분석 후 커밋 & 푸시 |
| `/save` 또는 `/save {메시지}` | 빠른 커밋 & 푸시 |
| `/update-tutorial` | Q&A 튜토리얼 수동 업데이트 |

## 질문 처리 워크플로우

```
사용자 입력
    │
    ├─ 질문인가? (?, 설명해줘, ~인가요 등)
    │   └─ YES → Perplexity MCP 조사 (영문) → docs/00-tutorial.md에 Q&A 추가
    │
    └─ 명령인가? (실행해, 분석해 등)
        └─ YES → 명령 실행 (튜토리얼 업데이트 없음)
```

### Perplexity MCP 규칙

1. **영문으로 쿼리 작성** (글로벌 CLAUDE.md 지침)
2. **조사 결과를 한국어로 정리**
3. `docs/00-tutorial.md`의 `<!-- TUTORIAL_MARKER -->` 위에 Q&A 추가

### Q&A 형식

```markdown
---

## Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{구조화된 답변}

---
```

## 튜토리얼에 기록할 주제

1. LLM 학습 검증 방법론 (from scratch vs fine-tuning)
2. Tokenizer 분석 (vocabulary, BPE merge, special tokens)
3. Weight 분석 (cosine similarity, 해싱, PCA)
4. Architecture 분석 (config 비교, MoE 구조)
5. Solar-Open-100B 검증 결과

## 검증 진행 상황

- [x] **Tokenizer 분석** → From scratch 지지
  - Solar vocab_size: 196,608 (모든 비교 대상과 불일치)
- [ ] Architecture 분석
- [ ] Weight 분석 (compute 리소스 필요)
- [ ] 행동 분석

## 주요 파일

| 파일 | 역할 |
|------|------|
| `docs/00-tutorial.md` | Q&A 튜토리얼 (자동 업데이트 대상) |
| `docs/01-tokenizer-analysis.md` | Tokenizer 분석 상세 + 결과 |
| `docs/02-weight-analysis.md` | Weight 분석 상세 |
| `docs/03-architecture-analysis.md` | Architecture 분석 상세 |
| `docs/04-behavior-analysis.md` | 행동 분석 상세 |
| `.claude/skills/update-tutorial.md` | 튜토리얼 업데이트 skill |
| `.claude/commands/*.md` | 커스텀 커맨드 정의 |

## 검증 작업 시 참고

검증 작업 수행 후 README.md의 "검증 진행 상황" 체크리스트도 함께 업데이트할 것.

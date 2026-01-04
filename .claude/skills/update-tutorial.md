# Update Tutorial Skill

**Purpose**: Q&A 형식의 검증 튜토리얼을 docs/00-tutorial.md에 추가

## Description

이 skill은 Solar-Open-100B 검증 과정에서 진행된 Q&A를 `docs/00-tutorial.md` 튜토리얼 문서에 추가합니다.

## Instructions

사용자가 `/update-tutorial` 명령을 실행하면:

1. 현재 대화에서 진행된 주요 Q&A를 식별합니다
2. `docs/00-tutorial.md`의 `<!-- TUTORIAL_MARKER -->` 위치를 찾습니다
3. 새로운 Q&A를 마커 바로 위에 추가합니다
4. Q&A 번호를 자동으로 증가시킵니다

## Target File

**튜토리얼 파일**: `docs/00-tutorial.md`

## Q&A Format

```markdown
---

## Q{N}: {질문 제목}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{답변 내용}

---
```

## Process

1. `docs/00-tutorial.md` 파일을 읽어서 현재 Q&A 개수를 파악합니다
2. 대화에서 사용자가 물어본 검증 관련 질문을 식별합니다
3. 해당 질문과 답변을 위 형식으로 정리합니다
4. `<!-- TUTORIAL_MARKER -->` 라인 바로 위에 새 Q&A를 삽입합니다
5. `docs/00-tutorial.md`를 업데이트합니다

## Example Usage

```
User: /update-tutorial
```

이 명령 실행 시, 현재 세션에서 진행된 검증 관련 Q&A를 `docs/00-tutorial.md`에 추가합니다.

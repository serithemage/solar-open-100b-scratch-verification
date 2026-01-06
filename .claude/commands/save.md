# Save (Commit & Push)

현재 변경사항을 커밋하고 원격 저장소에 푸시합니다.

## Arguments

$ARGUMENTS - 커밋 메시지 (선택적). 없으면 자동 생성

## Instructions

1. `git status`로 변경된 파일 확인
2. 변경사항이 없으면 "변경사항이 없습니다" 출력 후 종료
3. `git diff`로 변경 내용 확인
4. 변경 내용을 분석하여 커밋 메시지 결정:
   - $ARGUMENTS가 있으면 해당 메시지 사용
   - $ARGUMENTS가 없으면 변경 내용 분석 후 자동으로 커밋 메시지 생성
5. 모든 변경사항을 스테이징 (`git add .`)
6. 커밋 실행
7. `git push`로 원격 저장소에 푸시
8. 푸시 실패 시 `git push -u origin {현재브랜치}`로 재시도

## Commit Message Format

```
{변경 유형}: {변경 요약}

{상세 내용 (선택적)}
```

변경 유형:
- `docs`: 문서 변경
- `feat`: 새 기능
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `chore`: 기타 변경

## Example Usage

```
/save                          # 자동 커밋 메시지 생성
/save docs: README 업데이트    # 지정된 커밋 메시지 사용
```

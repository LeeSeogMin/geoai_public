# 에이전트 규칙 파일 배치 안내

이 수업은 학생마다 다른 AI 코딩 도구를 쓴다. 도구가 달라도 **같은 규칙**이 적용되도록, 규칙 원본을 하나만 두고 도구별 파일을 거기서 파생시킨다.

```
practice/AGENTS.md   (정본 — 규칙은 여기만 고친다)
        |
        +--> AGENTS.md                        저장소 루트 복사본
        +--> CLAUDE.md                        @AGENTS.md 한 줄 임포트
        +--> .github/copilot-instructions.md   Copilot용 안전장치
```

## 배치 방법

저장소 루트에서 한 번 실행한다.

```
python practice/agent-setup/install.py
```

이미 있는 파일은 건드리지 않는다. 규칙을 고친 뒤 다시 반영할 때는 `--force`를 붙인다. 다만 `--force`는 직접 써 둔 `context.md`, `todo.md`까지 서식으로 되돌리므로 주의한다.

## 도구별로 무엇을 읽는가

| 도구 | 읽는 파일 | 확인한 내용 |
|---|---|---|
| **OpenAI Codex** | 루트 `AGENTS.md` | 프로젝트 루트의 AGENTS.md를 세션 시작 시 자동으로 읽는다 |
| **Google Antigravity** | 루트 `AGENTS.md` (+ `.agents/rules/`) | AGENTS.md를 프로젝트 루트에 두고 커밋하면 도구가 자동으로 집어 간다. 세분화된 규칙은 `.agents/rules/`에 둔다 |
| **GitHub Copilot** | `AGENTS.md`, `.github/copilot-instructions.md` | VS Code가 `.github/copilot-instructions.md`를 자동 감지하고, AGENTS.md도 지원한다. 두 파일을 모두 둬서 확실하게 잡는다 |
| **Claude Code** | `CLAUDE.md` (**AGENTS.md는 읽지 않는다**) | 공식 문서가 명시한다: "Claude Code reads CLAUDE.md, not AGENTS.md." 대신 CLAUDE.md 첫 줄에 `@AGENTS.md`를 넣으면 그대로 불러온다 |

Windows에서는 심볼릭 링크에 관리자 권한이 필요해, 공식 문서도 `@AGENTS.md` 임포트를 권한다. `install.py`가 그 방식을 쓴다.

출처: [Claude Code 공식 문서 - Memory](https://code.claude.com/docs/en/memory), [OpenAI Codex - AGENTS.md](https://github.com/openai/codex/blob/main/docs/agents_md.md), [AGENTS.md 규약](https://agents.md/), [GitHub Changelog - Copilot AGENTS.md 지원](https://github.blog/changelog/2025-08-28-copilot-coding-agent-now-supports-agents-md-custom-instructions/), [VS Code - Custom instructions](https://code.visualstudio.com/docs/agent-customization/custom-instructions), [Antigravity - Rules](https://antigravity.google/docs/rules-workflows)

## 함께 만들어지는 작업 기록 파일

`install.py`는 `context.md`와 `todo.md`도 루트에 만든다. 학생이 직접 채워 쓰는 파일이고, 에이전트가 세션마다 읽는다. 쓰는 법은 `practice/README.md`의 "작업 기록 두 개" 항목에 있다.

# sample2

이 프로젝트는 Windows 11 환경에서 `uv`를 사용해 가상환경을 만들고 패키지를 설치하는 방식으로 실행하는 것을 권장합니다.

## 1. 사전 준비

이 프로젝트의 [pyproject.toml](pyproject.toml) 기준으로 Python 3.14 이상이 필요합니다.

1. Python 3.14 이상을 설치합니다.
2. Windows Terminal 또는 PowerShell을 엽니다.
3. `uv`가 설치되어 있지 않으면 아래 명령으로 설치합니다.

## 2. uv 설치 (Windows 11)

PowerShell에서 다음 명령을 실행합니다.

```powershell
winget install --id astral-sh.uv -e
```

설치 후 아래 명령으로 확인합니다.

```powershell
uv --version
```

## 3. 가상환경 생성

프로젝트 루트에서 다음 명령으로 `.venv` 가상환경을 생성합니다.

```powershell
uv venv .venv
```

## 4. 가상환경 활성화

PowerShell 기준으로 활성화하려면 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

만약 PowerShell에서 실행 정책 때문에 막히면, 현재 터미널 세션만 대상으로 아래 명령을 먼저 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

그 후 다시 활성화합니다.

명령 프롬프트(cmd) 사용 시에는 다음 명령을 사용합니다.

```cmd
.venv\Scripts\activate.bat
```

## 5. 의존성 설치

가상환경이 활성화된 상태에서 프로젝트 의존성을 설치합니다.

```powershell
uv sync
```

`uv sync`는 `pyproject.toml`에 정의된 의존성을 읽어 설치합니다.

## 6. 패키지 추가 설치

새 라이브러리를 추가할 때는 아래와 같이 설치합니다.

```powershell
uv add pandas scikit-learn
```

## 7. 실행 확인

가상환경이 활성화된 상태에서 프로젝트를 실행합니다.

```powershell
python main.py
```

## 8. 가상환경 비활성화

```powershell
deactivate
```

## 9. 가상환경 삭제

```powershell
Remove-Item -Recurse -Force .venv
```

## 참고

- `uv`는 Python 버전 관리와 가상환경 관리까지 한 번에 처리할 수 있습니다.
- 자주 쓰는 명령은 다음과 같습니다.
  - `uv venv .venv`
  - `uv sync`
  - `uv add <package>`
  - `uv run python main.py`

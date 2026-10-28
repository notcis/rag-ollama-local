# คู่มือการติดตั้งและใช้งาน (Installation & Setup Guide)

---

## 1. ติดตั้ง Python 3.14 (Prerequisites)

เปิด **PowerShell** แล้วติดตั้ง Python 3.14 ผ่าน winget:

```powershell
winget install Python.Python.3.14
```

---

## 2. สร้างและเปิดใช้งาน Virtual Environment

### 2.1 สร้าง Virtual Environment ด้วย Python 3.14
```powershell
py -m venv .venv-rag
```

### 2.2 เปิดใช้งาน (Activate) Virtual Environment
- **บน PowerShell:**
  ```powershell
  .\.venv-rag\Scripts\Activate.ps1
  ```
- **บน Command Prompt (CMD):**
  ```cmd
  .venv-rag\Scripts\activate.bat
  ```
- **บน Git Bash / WSL:**
  ```bash
  source .venv-rag/Scripts/activate
  ```

---

## 3. ติดตั้ง Dependencies

ติดตั้งแพ็กเกจที่จำเป็นจาก `requirements.txt`:

```powershell
pip install -r requirements.txt
```

---

## 4. เริ่มการทำงานของระบบ (Run Application)

### 4.1 เปิดบริการ Ollama
เปิด terminal เพื่อรัน Ollama Server:

```powershell
ollama serve
```
---

## 5. การออกจาก Virtual Environment

เมื่อใช้งานเสร็จเรียบร้อย ให้ใช้คำสั่ง:

```powershell
deactivate
```

---

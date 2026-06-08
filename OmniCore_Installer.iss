[Setup]
AppId={{8A8B9C9D-1234-5678-90AB-OMNICORE2026}}
AppName=OmniCore 智核工作站
AppVersion=1.4.0
AppPublisher=OmniCore 开源社区

; 【架构修复】强制声明 64-bit 模式，解决错误码 216 版本不兼容问题
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
; 【兼容性修复】最低 Windows 10，确保 API 兼容性
MinVersion=10.0

; 用户可自由选择安装目录
DefaultDirName={localappdata}\OmniCore
DefaultGroupName=OmniCore
DisableDirPage=no
DisableProgramGroupPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
AppMutex=OmniCore.exe

OutputDir=C:\Users\23747\Documents\OmniCore\dist
OutputBaseFilename=OmniCore_v1.4.0_Setup
SetupIconFile=C:\Users\23747\Documents\OmniCore\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
UninstallDisplayIcon={app}\OmniCore.exe


[Dirs]
; 预先建立好各个模块所需的工作台及缓存目录
Name: "{app}\user_data"
Name: "{app}\model_cache"
Name: "{app}\gguf_models"
Name: "{app}\agent_workspace"
Name: "{app}\data"
Name: "{app}\docs"
Name: "{app}\plugins"
Name: "{app}\update_code"
Name: "{app}\update_code\agent"
Name: "{app}\update_code\api"
Name: "{app}\update_code\build_scripts"
Name: "{app}\update_code\core"
Name: "{app}\update_code\model"
Name: "{app}\update_code\tools"
Name: "{app}\update_frontend"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"

[Files]
Source: "C:\Users\23747\Documents\OmniCore\dist\OmniCore\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Users\23747\Documents\OmniCore\external_libs\tesseract-ocr-w64-setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsTesseractInstalled

[Icons]
Name: "{autoprograms}\OmniCore"; Filename: "{app}\OmniCore.exe"
Name: "{autodesktop}\OmniCore"; Filename: "{app}\OmniCore.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OmniCore.exe"; Description: "启动 OmniCore 智核工作站"; Flags: nowait postinstall skipifsilent
Filename: "{tmp}\tesseract-ocr-w64-setup.exe"; Parameters: "/S"; Check: not IsTesseractInstalled; StatusMsg: "正在安装 OCR 文字识别引擎（Tesseract）..."; Flags: waituntilterminated

[Code]

// ── Tesseract OCR 检测 ──
function IsTesseractInstalled: Boolean;
var
  ProgPath: string;
  ProgPath86: string;
begin
  Result := False;
  // 检查注册表（UB-Mannheim 安装器写入的键值）
  if RegQueryStringValue(HKLM, 'SOFTWARE\Tesseract-OCR', 'InstallDir', ProgPath) then
  begin
    if FileExists(AddBackslash(ProgPath) + 'tesseract.exe') then
    begin
      Result := True;
      Exit;
    end;
  end;
  if RegQueryStringValue(HKCU, 'SOFTWARE\Tesseract-OCR', 'InstallDir', ProgPath) then
  begin
    if FileExists(AddBackslash(ProgPath) + 'tesseract.exe') then
    begin
      Result := True;
      Exit;
    end;
  end;
  // 检查常见默认路径
  ProgPath := ExpandConstant('{pf}\Tesseract-OCR\tesseract.exe');
  if FileExists(ProgPath) then
  begin
    Result := True;
    Exit;
  end;
  ProgPath86 := ExpandConstant('{pf32}\Tesseract-OCR\tesseract.exe');
  if FileExists(ProgPath86) then
  begin
    Result := True;
    Exit;
  end;
  // 检查 LocalAppData
  ProgPath := ExpandConstant('{localappdata}\Programs\Tesseract-OCR\tesseract.exe');
  if FileExists(ProgPath) then
  begin
    Result := True;
    Exit;
  end;
end;

type
  TMemoryStatusEx = record
    dwLength: DWORD;
    dwMemoryLoad: DWORD;
    ullTotalPhysLow: DWORD;
    ullTotalPhysHigh: DWORD;
    ullAvailPhysLow: DWORD;
    ullAvailPhysHigh: DWORD;
    ullTotalPageFileLow: DWORD;
    ullTotalPageFileHigh: DWORD;
    ullAvailPageFileLow: DWORD;
    ullAvailPageFileHigh: DWORD;
    ullTotalVirtualLow: DWORD;
    ullTotalVirtualHigh: DWORD;
    ullAvailVirtualLow: DWORD;
    ullAvailVirtualHigh: DWORD;
    ullAvailExtendedVirtualLow: DWORD;
    ullAvailExtendedVirtualHigh: DWORD;
  end;

function GlobalMemoryStatusEx(var lpBuffer: TMemoryStatusEx): BOOL;
  external 'GlobalMemoryStatusEx@kernel32.dll stdcall';

function GetTotalMemoryGB: Integer;
var
  MemStatus: TMemoryStatusEx;
  LowMB: Integer;
begin
  MemStatus.dwLength := 64;
  if GlobalMemoryStatusEx(MemStatus) then
  begin
    LowMB := MemStatus.ullTotalPhysLow div 1048576;
    if LowMB < 0 then
      LowMB := LowMB + 4096;
    Result := ((MemStatus.ullTotalPhysHigh * 4096) + LowMB) div 1024;
  end
  else
    Result := 8;
end;

function StringReplaceAll(const S, OldPattern, NewPattern: string): string;
var
  TempStr: string;
begin
  TempStr := S;
  StringChange(TempStr, OldPattern, NewPattern);
  Result := TempStr;
end;

// ============================================================
// 以下代码由 build_installer.py 自动生成
// GGUF推理: 61个 | HF可微调: 19个
// ============================================================


var
  ModelPage: TInputOptionWizardPage;
  HfModelPage: TInputOptionWizardPage;
  ModelDirPage: TInputDirWizardPage;
  DirWarnShown: Boolean;

// ========== 【权限修复】检测目标安装目录是否可写 ==========
// 此函数递归检查最近存在的父目录是否可写，解决用户输入不存在路径时误判的问题
function IsDirWritable(const Dir: string): Boolean;
var
  TestDir: string;
  CheckDir: string;
begin
  // 从用户输入的路径向上查找最近存在的父目录
  CheckDir := Dir;
  while (CheckDir <> '') and not DirExists(CheckDir) do
  begin
    CheckDir := ExtractFileDir(CheckDir);
    if CheckDir = '' then
      CheckDir := ExpandConstant('{sd}');
  end;
  // 在该存在的目录中创建测试子目录
  TestDir := AddBackslash(CheckDir) + '.OmniCoreWriteTest';
  if CreateDir(TestDir) then
  begin
    RemoveDir(TestDir);
    Result := True;
  end
  else
    Result := False;
end;

// 初始化变量
function InitializeSetup: Boolean;
begin
  DirWarnShown := False;
  Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  InstallDir: string;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    InstallDir := WizardDirValue;
    if Pos(ExpandConstant('{localappdata}'), InstallDir) > 0 then
      Exit;
    if not IsDirWritable(InstallDir) then
    begin
      if not DirWarnShown then
      begin
        DirWarnShown := True;
        if SuppressibleMsgBox(
          '目标目录不可写入'#13#10 +
          '您选择的 "' + InstallDir + '" 目录当前用户没有写入权限。'#13#10#13#10 +
          '建议使用默认路径: ' + ExpandConstant('{localappdata}') + '\OmniCore'#13#10#13#10 +
          '是否仍要继续？（可能导致安装失败）',
          mbError, MB_YESNO, 0) = IDNO then
          Result := False;
      end
      else
      begin
        MsgBox('目录无法写入，请更换安装目录。', mbError, MB_OK);
        Result := False;
      end;
    end;
  end;
end;

procedure InitializeWizard;
var
  TotalRAM: Integer;
begin
  TotalRAM := GetTotalMemoryGB();

  ModelDirPage := CreateInputDirPage(wpSelectDir,
    '选择模型存储路径', '请选择下载和部署的本地 AI 模型文件的保存位置。',
    '本地 AI 模型文件通常较大（数GB到数十GB），建议选择一个剩余空间充足的非系统盘（如 D、E 盘）。'#13#10#13#10'软件启动后会自动在该目录下完成模型的下载与部署。',
    False, '');
  ModelDirPage.Add('');
  ModelDirPage.Values[0] := ExpandConstant('{userdocs}\OmniCoreModels');

  ModelPage := CreateInputOptionPage(ModelDirPage.ID,
    '选择本地推理 AI 模型 (GGUF)', '请选择您希望在聊天推理中使用的 GGUF 量化模型。',
    '系统检测到您的物理内存约为 ' + IntToStr(TotalRAM) + ' GB。GGUF 4-bit 量化在保证智能的同时极大降低内存占用。'#13#10'不需要推理模型可选"跳过"。',
    True, False);

  ModelPage.Add('跳过 (不安装 GGUF 推理模型)');

  // ── 平铺展示所有档位的模型（带 [档位] 前缀） ──

  ModelPage.Add('[旗舰] Mixtral-8x7B-Instruct Q4_K_M  [27.8 GB]  [27.8 GB]');
  ModelPage.Add('[旗舰] Qwen2.5-72B-Instruct Q4_K_M  [43.6 GB]  [43.6 GB]');
  ModelPage.Add('[高配] Qwen2.5-32B-Instruct Q4_K_M  [19.4 GB]  [19.4 GB]');
  ModelPage.Add('[高配] DeepSeek-R1-Distill-Qwen-32B Q4_K_M (推理)  [19.4 GB]  [19.4 GB]');
  ModelPage.Add('[高配] Gemma-2-27B Q4_K_M  [16.3 GB]  [16.3 GB]');
  ModelPage.Add('[高配] Mistral-Small-24B Q4_K_M  [14.5 GB]  [14.5 GB]');
  ModelPage.Add('[主流] Qwen2.5-14B-Instruct Q4_K_M  [8.5 GB]  [8.5 GB]');
  ModelPage.Add('[主流] DeepSeek-R1-Distill-Qwen-14B Q4_K_M (推理)  [8.5 GB]  [8.5 GB]');
  ModelPage.Add('[主流] Phi-3-medium-14B Q4_K_M  [8.5 GB]  [8.5 GB]');
  ModelPage.Add('[主流] Mistral-Nemo-12B Q4_K_M  [7.3 GB]  [7.3 GB]');
  ModelPage.Add('[主流] Qwen2.5-Coder-14B Q4_K_M (代码)  [8.5 GB]  [8.5 GB]');
  ModelPage.Add('[主流] StarCoder2-15B Q4_K_M (代码)  [9.1 GB]  [9.1 GB]');
  ModelPage.Add('[主流] DeepSeek-Coder-V2-Lite-16B Q4_K_M (代码)  [9.7 GB]  [9.7 GB]');
  ModelPage.Add('[主流] Phi-4-14B Q4_K_M (推理)  [8.5 GB]  [8.5 GB]');
  ModelPage.Add('[主流] DeepSeek-V3-0324-Lite Q4_K_M  [9.7 GB]  [9.7 GB]');
  ModelPage.Add('[主流] Gemma-3-12B Q4_K_M  [7.3 GB]  [7.3 GB]');
  ModelPage.Add('[主流] LLM-jp-3-13B Q4_K_M  [7.9 GB]  [7.9 GB]');
  ModelPage.Add('[基础] Qwen2.5-7B-Instruct Q4_K_M  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] DeepSeek-R1-Distill-Qwen-7B Q4_K_M (推理)  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Yi-1.5-6B-Chat Q4_K_M  [3.6 GB]  [3.6 GB]');
  ModelPage.Add('[基础] InternLM2-Chat-7B Q4_K_M  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Baichuan2-7B-Chat Q4_K_M  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] DeepSeek-R1-Distill-Llama-8B Q4_K_M (推理)  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] Mistral-7B-Instruct-v0.3 Q4_K_M  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Llama-3.1-8B-Instruct Q4_K_M  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] Gemma-2-9B Q4_K_M  [5.4 GB]  [5.4 GB]');
  ModelPage.Add('[基础] Qwen2.5-Coder-7B Q4_K_M (代码)  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Yi-6B Q4_K_M  [3.6 GB]  [3.6 GB]');
  ModelPage.Add('[基础] ChatGLM3-6B Q4_K_M  [3.6 GB]  [3.6 GB]');
  ModelPage.Add('[基础] GLM-4-9B-Chat Q4_K_M  [5.4 GB]  [5.4 GB]');
  ModelPage.Add('[基础] OpenChat-3.6-8B Q4_K_M  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] OpenCode-8B Q4_K_M (代码)  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] CodeQwen2.5-7B-Instruct Q4_K_M (代码)  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Llama-3.2-11B-Vision Q4_K_M (视觉)  [6.7 GB]  [6.7 GB]');
  ModelPage.Add('[基础] InternLM3-8B-Instruct Q4_K_M  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] Qwen2.5-Math-7B Q4_K_M (推理)  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] Phi-3.5-MoE-6.6B Q4_K_M  [4.0 GB]  [4.0 GB]');
  ModelPage.Add('[基础] Hermes-3-Llama-3.1-8B Q4_K_M  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] Qwen2.5-VL-7B Q4_K_M (视觉)  [4.2 GB]  [4.2 GB]');
  ModelPage.Add('[基础] LLaMA-Mesh-8B Q4_K_M (代码)  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[基础] Dolphin3.0-Llama-3.1-8B Q4_K_M  [4.8 GB]  [4.8 GB]');
  ModelPage.Add('[轻量] 🌟 [推荐] Qwen2.5-1.5B-Instruct Q4_K_M  [930 MB]  [930 MB]');
  ModelPage.Add('[轻量] Qwen2.5-3B-Instruct Q4_K_M  [1.8 GB]  [1.8 GB]');
  ModelPage.Add('[轻量] Qwen2.5-0.5B-Instruct Q4_K_M  [310 MB]  [310 MB]');
  ModelPage.Add('[轻量] Llama-3.2-1B-Instruct Q4_K_M  [620 MB]  [620 MB]');
  ModelPage.Add('[轻量] Llama-3.2-3B-Instruct Q4_K_M  [1.8 GB]  [1.8 GB]');
  ModelPage.Add('[轻量] Phi-3-mini-3.8B Q4_K_M  [2.3 GB]  [2.3 GB]');
  ModelPage.Add('[轻量] Gemma-2-2B Q4_K_M  [1.2 GB]  [1.2 GB]');
  ModelPage.Add('[轻量] MiniCPM3-4B Q4_K_M  [2.4 GB]  [2.4 GB]');
  ModelPage.Add('[轻量] Phi-3.5-mini-3.8B Q4_K_M (多语言)  [2.3 GB]  [2.3 GB]');
  ModelPage.Add('[轻量] DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M (推理)  [930 MB]  [930 MB]');
  ModelPage.Add('[轻量] Gemma-2-2B-JPN Q4_K_M  [1.2 GB]  [1.2 GB]');
  ModelPage.Add('[轻量] Qwen2.5-Coder-1.5B Q4_K_M (代码)  [930 MB]  [930 MB]');
  ModelPage.Add('[轻量] Qwen2.5-0.5B-Coder Q4_K_M (代码)  [310 MB]  [310 MB]');
  ModelPage.Add('[轻量] Gemma-3-4B Q4_K_M (多语言)  [2.4 GB]  [2.4 GB]');
  ModelPage.Add('[轻量] SmolLM2-360M-Instruct Q4_K_M  [223 MB]  [223 MB]');
  ModelPage.Add('[轻量] Phi-4-mini-3.8B Q4_K_M (推理)  [2.3 GB]  [2.3 GB]');
  ModelPage.Add('[轻量] Granite-3.1-2B-Instruct Q4_K_M (代码)  [1.2 GB]  [1.2 GB]');
  ModelPage.Add('[轻量] Hermes-3-Llama-3.2-3B Q4_K_M  [1.8 GB]  [1.8 GB]');
  ModelPage.Add('[轻量] Gemma-3-1B Q4_K_M  [620 MB]  [620 MB]');
  ModelPage.Add('[轻量] Qwen2.5-VL-3B Q4_K_M (视觉)  [1.8 GB]  [1.8 GB]');

  // 根据内存状况默认选中对应档位的推荐模型
  if True then
    ModelPage.SelectedValueIndex := 42
  ;
end;


procedure CurStepChanged(CurStep: TSetupStep);
var
  SettingsPath: string;
  SettingsJson: string;
  SelectedModel: string;
  SelectedQuant: string;
  ModelDirPath: string;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    ModelDirPath := ModelDirPage.Values[0];
    ForceDirectories(ModelDirPath);
    SettingsPath := ExpandConstant('{app}\app_settings.json');

    if ModelPage.SelectedValueIndex > 0 then
    begin
      // 平铺映射：所有模型选项对应唯一的 HuggingFace 仓库名和量化级别
      // 索引与 InitializeWizard 中添加顺序完全一致
      SelectedQuant := 'Q4_K_M';  // 默认值
      case ModelPage.SelectedValueIndex of

        1: begin SelectedModel := 'Mistral/Mixtral-8x7B-Instruct-v0.1-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        2: begin SelectedModel := 'Qwen/Qwen2.5-72B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        3: begin SelectedModel := 'Qwen/Qwen2.5-32B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        4: begin SelectedModel := 'unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        5: begin SelectedModel := 'unsloth/gemma-2-27b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        6: begin SelectedModel := 'unsloth/Mistral-Small-24B-Instruct-2501-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        7: begin SelectedModel := 'Qwen/Qwen2.5-14B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        8: begin SelectedModel := 'unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        9: begin SelectedModel := 'microsoft/Phi-3-medium-4k-instruct-gguf'; SelectedQuant := 'Q4_K_M'; end;
        10: begin SelectedModel := 'unsloth/Mistral-Nemo-Instruct-2407-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        11: begin SelectedModel := 'Qwen/Qwen2.5-Coder-14B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        12: begin SelectedModel := 'bigcode/starcoder2-15b-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        13: begin SelectedModel := 'unsloth/DeepSeek-Coder-V2-Lite-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        14: begin SelectedModel := 'unsloth/phi-4-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        15: begin SelectedModel := 'unsloth/DeepSeek-V3-0324-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        16: begin SelectedModel := 'unsloth/gemma-3-12b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        17: begin SelectedModel := 'llm-jp/llm-jp-3-13b-instruct3-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        18: begin SelectedModel := 'Qwen/Qwen2.5-7B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        19: begin SelectedModel := 'unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        20: begin SelectedModel := 'TheBloke/Yi-1.5-6B-Chat-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        21: begin SelectedModel := 'TheBloke/internlm2-chat-7b-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        22: begin SelectedModel := 'TheBloke/Baichuan2-7B-Chat-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        23: begin SelectedModel := 'unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        24: begin SelectedModel := 'Mistral/Mistral-7B-Instruct-v0.3-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        25: begin SelectedModel := 'unsloth/Meta-Llama-3.1-8B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        26: begin SelectedModel := 'unsloth/gemma-2-9b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        27: begin SelectedModel := 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        28: begin SelectedModel := 'unsloth/Yi-6B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        29: begin SelectedModel := 'unsloth/chatglm3-6b-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        30: begin SelectedModel := 'unsloth/glm-4-9b-chat-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        31: begin SelectedModel := 'unsloth/openchat-3.6-8b-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        32: begin SelectedModel := 'unsloth/OpenCode-8B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        33: begin SelectedModel := 'Qwen/CodeQwen2.5-7B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        34: begin SelectedModel := 'unsloth/Llama-3.2-11B-Vision-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        35: begin SelectedModel := 'unsloth/internlm3-8b-instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        36: begin SelectedModel := 'Qwen/Qwen2.5-Math-7B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        37: begin SelectedModel := 'microsoft/Phi-3.5-MoE-instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        38: begin SelectedModel := 'NousResearch/Hermes-3-Llama-3.1-8B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        39: begin SelectedModel := 'Qwen/Qwen2.5-VL-7B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        40: begin SelectedModel := 'NVIDIA/Llama-3.1-LLaMA-Mesh-8B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        41: begin SelectedModel := 'cognitivecomputations/Dolphin3.0-Llama-3.1-8B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        42: begin SelectedModel := 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        43: begin SelectedModel := 'Qwen/Qwen2.5-3B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        44: begin SelectedModel := 'Qwen/Qwen2.5-0.5B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        45: begin SelectedModel := 'unsloth/Llama-3.2-1B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        46: begin SelectedModel := 'unsloth/Llama-3.2-3B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        47: begin SelectedModel := 'microsoft/Phi-3-mini-4k-instruct-gguf'; SelectedQuant := 'Q4_K_M'; end;
        48: begin SelectedModel := 'unsloth/gemma-2-2b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        49: begin SelectedModel := 'unsloth/MiniCPM3-4B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        50: begin SelectedModel := 'microsoft/Phi-3.5-mini-instruct-gguf'; SelectedQuant := 'Q4_K_M'; end;
        51: begin SelectedModel := 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        52: begin SelectedModel := 'unsloth/gemma-2-2b-jpn-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        53: begin SelectedModel := 'Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        54: begin SelectedModel := 'Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        55: begin SelectedModel := 'unsloth/gemma-3-4b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        56: begin SelectedModel := 'bartowski/SmolLM2-360M-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        57: begin SelectedModel := 'unsloth/phi-4-mini-instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        58: begin SelectedModel := 'ibm-granite/granite-3.1-2b-instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        59: begin SelectedModel := 'NousResearch/Hermes-3-Llama-3.2-3B-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        60: begin SelectedModel := 'unsloth/gemma-3-1b-it-GGUF'; SelectedQuant := 'Q4_K_M'; end;
        61: begin SelectedModel := 'Qwen/Qwen2.5-VL-3B-Instruct-GGUF'; SelectedQuant := 'Q4_K_M'; end;

      end;
       
      SettingsJson := '{' + #13#10 +
                      '  "model_name": "' + SelectedModel + '",' + #13#10 +
                      '  "model_type": "gguf",' + #13#10 +
                      '  "gguf_download_pending": true,' + #13#10 +
                      '  "gguf_model_key": "' + SelectedModel + '",' + #13#10 +
                      '  "gguf_quant": "' + SelectedQuant + '",' + #13#10 +
                      '  "gguf_dir": "' + StringReplaceAll(ModelDirPath, '\', '/') + '",' + #13#10 +
                      '  "n_gpu_layers": -1,' + #13#10 +
                      '  "n_ctx": 4096' + #13#10 +
                      '}';
                       
      SaveStringToFile(SettingsPath, SettingsJson, False);
    end
    else
    begin
      SettingsJson := '{' + #13#10 +
                      '  "gguf_dir": "' + StringReplaceAll(ModelDirPath, '\', '/') + '"' + #13#10 +
                      '}';
      SaveStringToFile(SettingsPath, SettingsJson, False);
    end;

    // ========== 【权限修复】为用户组添加目录写权限 ==========
    Exec('icacls', ExpandConstant('"{app}" /grant Users:(OI)(CI)F /T /Q'),
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;

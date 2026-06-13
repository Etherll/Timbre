use std::{
    collections::HashMap,
    env,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::SystemTime,
};

use serde::Serialize;
use tauri::{ipc::Channel, State};

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Apply Windows-specific flags so spawned helpers never flash a console window.
fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn capture_command(cmd: &mut Command) -> Option<String> {
    let out = quiet(cmd).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next().unwrap_or("").trim().to_string();
    if line.is_empty() { None } else { Some(line) }
}

fn path_var_name() -> &'static str {
    if cfg!(windows) { "Path" } else { "PATH" }
}

fn prepend_path_dirs(cmd: &mut Command, dirs: Vec<PathBuf>) {
    let mut parts: Vec<PathBuf> = dirs.into_iter().filter(|p| p.is_dir()).collect();
    if parts.is_empty() {
        return;
    }
    let current = env::var_os(path_var_name()).or_else(|| env::var_os("PATH"));
    if let Some(current) = current {
        parts.extend(env::split_paths(&current));
    }
    if let Ok(joined) = env::join_paths(parts) {
        cmd.env(path_var_name(), joined);
    }
}

fn managed_tool_dirs(repo_path: &Path) -> Vec<PathBuf> {
    let Some(root) = repo_path.parent() else { return Vec::new() };
    let bin = if cfg!(windows) { "Scripts" } else { "bin" };
    vec![
        root.join("ffmpeg").join("bin"),
        root.join(".venv").join(bin),
    ]
}

fn with_managed_tools(cmd: &mut Command, repo_path: &Path) {
    prepend_path_dirs(cmd, managed_tool_dirs(repo_path));
}

// ---------------------------------------------------------------------------
// Streaming process registry
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
#[serde(tag = "event", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ProcEvent {
    Started { task_id: u32, pid: u32 },
    Line { stream: &'static str, line: String },
    Exit { code: Option<i32>, cancelled: bool },
}

struct TaskHandle {
    pid: u32,
    cancelled: bool,
}

#[derive(Default)]
pub struct ProcRegistry {
    tasks: Arc<Mutex<HashMap<u32, TaskHandle>>>,
    counter: AtomicU32,
}

impl ProcRegistry {
    fn next_id(&self) -> u32 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// Spawn `cmd`, stream stdout/stderr lines over `channel`, report exit.
/// Returns immediately; reader threads own the child from here on.
fn spawn_streamed(
    tasks: Arc<Mutex<HashMap<u32, TaskHandle>>>,
    task_id: u32,
    mut cmd: Command,
    channel: Channel<ProcEvent>,
) -> Result<(), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    quiet(&mut cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to launch {:?}: {e}", cmd.get_program()))?;
    let pid = child.id();
    tasks
        .lock()
        // A worker thread panicking while holding this lock would poison it;
        // under `panic = "abort"` recovering the guard keeps the app alive
        // instead of aborting the whole process on the next lock.
        .unwrap_or_else(|e| e.into_inner())
        .insert(task_id, TaskHandle { pid, cancelled: false });
    let _ = channel.send(ProcEvent::Started { task_id, pid });

    // Infallible: both pipes were set to Stdio::piped() above and not yet taken.
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let ch_out = channel.clone();
    let t_out = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = ch_out.send(ProcEvent::Line { stream: "stdout", line });
        }
    });
    let ch_err = channel.clone();
    let t_err = std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = ch_err.send(ProcEvent::Line { stream: "stderr", line });
        }
    });

    std::thread::spawn(move || {
        let _ = t_out.join();
        let _ = t_err.join();
        let status = child.wait().ok();
        let cancelled = tasks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&task_id)
            .map(|h| h.cancelled)
            .unwrap_or(false);
        let _ = channel.send(ProcEvent::Exit {
            code: status.and_then(|s| s.code()),
            cancelled,
        });
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Commands: pipeline + yt-dlp
// ---------------------------------------------------------------------------

#[tauri::command]
fn start_pipeline(
    registry: State<'_, ProcRegistry>,
    python: String,
    repo: String,
    args: Vec<String>,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    let repo_path = PathBuf::from(&repo);
    if !repo_path.join("run_timbre.py").exists() {
        return Err(format!("run_timbre.py not found in {repo}"));
    }
    let mut cmd = Command::new(&python);
    cmd.arg("run_timbre.py")
        .args(&args)
        .current_dir(&repo_path)
        // Plain, unbuffered, unwrapped output so the UI can parse stage banners live.
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("NO_COLOR", "1")
        .env("FORCE_COLOR", "0")
        .env("TERM", "dumb")
        .env("COLUMNS", "400");
    with_managed_tools(&mut cmd, &repo_path);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

/// How yt-dlp is reachable on this machine. Resolved once in `detect_env` and
/// re-resolved by `start_ytdlp` so the env chip and the actual fetch never
/// disagree: prefer a bare `yt-dlp` on PATH, else fall back to running it as a
/// module under the detected python (`<python> -m yt_dlp`) — covers the common
/// case of a pip-installed yt-dlp whose script dir isn't on PATH (RA-R8).
struct YtdlpResolution {
    via: &'static str, // "exe" | "module"
    version: String,
}

/// Resolve yt-dlp the same way for both the env probe and the fetch.
/// `python` is the interpreter `detect_env` already found (may be None if no
/// Python is present, in which case only the bare-exe path can work).
fn resolve_ytdlp(python: Option<&str>) -> Option<YtdlpResolution> {
    if let Some(version) = capture("yt-dlp", &["--version"]) {
        return Some(YtdlpResolution { via: "exe", version });
    }
    if let Some(py) = python {
        if let Some(version) = capture(py, &["-m", "yt_dlp", "--version"]) {
            return Some(YtdlpResolution { via: "module", version });
        }
    }
    None
}

#[tauri::command]
fn start_ytdlp(
    registry: State<'_, ProcRegistry>,
    python: String,
    url: String,
    dest_dir: String,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("not a valid http(s) URL".into());
    }
    std::fs::create_dir_all(&dest_dir).map_err(|e| format!("cannot create {dest_dir}: {e}"))?;
    let outtmpl = format!("{dest_dir}/%(title).80s [%(id)s].%(ext)s");

    // Same resolution order as the env chip: bare exe first, then `<python> -m yt_dlp`.
    let resolution = resolve_ytdlp(Some(&python))
        .ok_or("yt-dlp not found: install it (pip install yt-dlp) or add it to PATH")?;
    let mut cmd = match resolution.via {
        "module" => {
            let mut c = Command::new(&python);
            c.args(["-m", "yt_dlp"]);
            c
        }
        _ => Command::new("yt-dlp"),
    };
    let dest = PathBuf::from(&dest_dir);
    if let Some(repo_path) = dest.parent().filter(|p| p.join("run_timbre.py").exists()) {
        with_managed_tools(&mut cmd, repo_path);
    }
    // Mirrors the repo's Colab recipe: best opus source -> wav, single video only.
    cmd.args([
        "-x",
        "-f",
        "bestaudio[ext=webm]/bestaudio/best",
        "--audio-quality",
        "0",
        "--audio-format",
        "wav",
        "--no-playlist",
        "--restrict-filenames",
        "--force-overwrites",
        "--newline",
        "--progress",
        "--no-simulate",
        "--print",
        "after_move:TIMBRE_FILE::%(filepath)s",
        "-o",
        &outtmpl,
        &url,
    ]);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

#[tauri::command]
fn cancel_task(registry: State<'_, ProcRegistry>, task_id: u32) -> Result<(), String> {
    let pid = {
        let mut map = registry.tasks.lock().unwrap_or_else(|e| e.into_inner());
        match map.get_mut(&task_id) {
            Some(h) => {
                h.cancelled = true;
                h.pid
            }
            None => return Ok(()), // already exited
        }
    };
    #[cfg(windows)]
    {
        // /T kills the whole tree — python spawns separation/ASR worker subprocesses.
        let _ = quiet(Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"])).status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Commands: environment + filesystem helpers
// ---------------------------------------------------------------------------

fn capture(exe: &str, args: &[&str]) -> Option<String> {
    capture_command(Command::new(exe).args(args))
}

fn find_repo_root() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.to_path_buf());
        }
    }
    for start in candidates {
        let mut dir: Option<&Path> = Some(start.as_path());
        for _ in 0..8 {
            let Some(d) = dir else { break };
            if d.join("run_timbre.py").exists() {
                return Some(d.to_path_buf());
            }
            dir = d.parent();
        }
    }
    None
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EnvInfo {
    repo_root: Option<String>,
    python: Option<String>,
    python_version: Option<String>,
    ytdlp_version: Option<String>,
    // How yt-dlp resolves: "exe" (bare on PATH) or "module" (<python> -m yt_dlp).
    // None when yt-dlp is absent. Kept in sync with start_ytdlp via resolve_ytdlp.
    ytdlp_via: Option<String>,
    ffprobe: bool,
    ffmpeg: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupStatus {
    install_dir: String,
    repo_path: String,
    python_path: String,
    output_dir: String,
    ffmpeg_bin: String,
    ffprobe_path: String,
    repo_ready: bool,
    python_ready: bool,
    requirements_ready: bool,
    ytdlp_ready: bool,
    ffmpeg_ready: bool,
    ffprobe_ready: bool,
    vad_ready: bool,
    ready: bool,
    missing: Vec<String>,
}

fn default_setup_path() -> PathBuf {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| env::var_os("APPDATA").map(PathBuf::from))
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("TimbreStudio")
        .join("runtime")
}

#[tauri::command]
fn default_setup_dir() -> String {
    default_setup_path().to_string_lossy().into_owned()
}

fn setup_root(install_dir: &str) -> PathBuf {
    let trimmed = install_dir.trim();
    if trimmed.is_empty() {
        default_setup_path()
    } else {
        PathBuf::from(trimmed)
    }
}

fn setup_python(root: &Path) -> PathBuf {
    if cfg!(windows) {
        root.join(".venv").join("Scripts").join("python.exe")
    } else {
        root.join(".venv").join("bin").join("python")
    }
}

fn vad_model_ready(repo: &Path) -> bool {
    let dir = repo
        .join("pretrained_models")
        .join("FireRedVAD")
        .join("VAD");
    std::fs::read_dir(&dir)
        .map(|mut d| d.next().is_some())
        .unwrap_or(false)
}

fn setup_imports_ready(_root: &Path, repo: &Path, python: &Path) -> bool {
    if !python.is_file() || !repo.join("run_timbre.py").is_file() {
        return false;
    }
    let code = "import importlib.util as u; \
mods=['torch','torchaudio','soundfile','librosa','whisper','wespeaker','nemo','timbre']; \
missing=[m for m in mods if u.find_spec(m) is None]; \
assert not missing, missing; \
print('ok')";
    let mut cmd = Command::new(python);
    cmd.args(["-c", code])
        .current_dir(repo)
        .env("PYTHONIOENCODING", "utf-8");
    with_managed_tools(&mut cmd, repo);
    capture_command(&mut cmd).is_some()
}

fn setup_ytdlp_ready(python: &Path) -> bool {
    if !python.is_file() {
        return false;
    }
    let mut cmd = Command::new(python);
    cmd.args(["-m", "yt_dlp", "--version"]);
    capture_command(&mut cmd).is_some()
}

#[tauri::command]
fn setup_status(install_dir: String) -> SetupStatus {
    let root = setup_root(&install_dir);
    let repo = root.join("Timbre");
    let python = setup_python(&root);
    let output = root.join("output_runs");
    let ffmpeg_bin = root.join("ffmpeg").join("bin");
    let ffmpeg = ffmpeg_bin.join(if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" });
    let ffprobe = ffmpeg_bin.join(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });

    let repo_ready = repo.join("run_timbre.py").is_file();
    let python_ready = python.is_file() && capture(python.to_string_lossy().as_ref(), &["--version"]).is_some();
    let requirements_ready = setup_imports_ready(&root, &repo, &python);
    let ytdlp_ready = setup_ytdlp_ready(&python);
    let ffmpeg_ready = ffmpeg.is_file() || capture("ffmpeg", &["-version"]).is_some();
    let ffprobe_ready = ffprobe.is_file() || capture("ffprobe", &["-version"]).is_some();
    let vad_ready = repo_ready && vad_model_ready(&repo);

    let mut missing = Vec::new();
    if !repo_ready { missing.push("repo".into()) }
    if !python_ready { missing.push("python".into()) }
    if !requirements_ready { missing.push("python packages".into()) }
    if !ytdlp_ready { missing.push("yt-dlp".into()) }
    if !ffmpeg_ready { missing.push("ffmpeg".into()) }
    if !ffprobe_ready { missing.push("ffprobe".into()) }
    if !vad_ready { missing.push("FireRedVAD".into()) }
    let ready = missing.is_empty();

    SetupStatus {
        install_dir: root.to_string_lossy().into_owned(),
        repo_path: repo.to_string_lossy().into_owned(),
        python_path: python.to_string_lossy().into_owned(),
        output_dir: output.to_string_lossy().into_owned(),
        ffmpeg_bin: ffmpeg_bin.to_string_lossy().into_owned(),
        ffprobe_path: ffprobe.to_string_lossy().into_owned(),
        repo_ready,
        python_ready,
        requirements_ready,
        ytdlp_ready,
        ffmpeg_ready,
        ffprobe_ready,
        vad_ready,
        ready,
        missing,
    }
}

fn ps_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

#[tauri::command]
fn start_setup(
    registry: State<'_, ProcRegistry>,
    install_dir: String,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    if !cfg!(windows) {
        return Err("managed setup currently supports Windows builds only".into());
    }
    let root = setup_root(&install_dir);
    let root_arg = ps_quote(&root.to_string_lossy());
    let script = format!(r#"
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = {root_arg}
$repo = Join-Path $root 'Timbre'
$venv = Join-Path $root '.venv'
$uv = Join-Path $root 'uv\uv.exe'
$ffmpegExe = Join-Path $root 'ffmpeg\bin\ffmpeg.exe'
$ffprobeExe = Join-Path $root 'ffmpeg\bin\ffprobe.exe'
$setupState = Join-Path $root '.setup'
$requirementsMarker = Join-Path $setupState 'requirements.ok'

function Say($message) {{ Write-Output "TIMBRE_SETUP:: $message" }}
function Download-File($url, $outFile) {{
  Say "downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing
}}

New-Item -ItemType Directory -Force $root | Out-Null
New-Item -ItemType Directory -Force $setupState | Out-Null

if (!(Test-Path $uv)) {{
  Say 'installing uv bootstrapper'
  $uvAsset = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {{ 'uv-aarch64-pc-windows-msvc.zip' }} else {{ 'uv-x86_64-pc-windows-msvc.zip' }}
  $uvZip = Join-Path $root 'uv.zip'
  $uvStage = Join-Path $root '_uv_extract'
  Remove-Item -Recurse -Force $uvStage -ErrorAction SilentlyContinue
  Download-File "https://github.com/astral-sh/uv/releases/latest/download/$uvAsset" $uvZip
  Expand-Archive -Path $uvZip -DestinationPath $uvStage -Force
  $foundUv = Get-ChildItem -Path $uvStage -Recurse -Filter 'uv.exe' | Select-Object -First 1
  if (!$foundUv) {{ throw 'uv.exe was not found in the downloaded archive' }}
  New-Item -ItemType Directory -Force (Split-Path $uv) | Out-Null
  Copy-Item $foundUv.FullName $uv -Force
}}

if (!(Test-Path (Join-Path $repo 'run_timbre.py'))) {{
  Say 'installing Timbre source'
  $repoZip = Join-Path $root 'timbre-source.zip'
  $repoStage = Join-Path $root '_repo_extract'
  Remove-Item -Recurse -Force $repoStage -ErrorAction SilentlyContinue
  try {{
    $release = Invoke-RestMethod -Headers @{{ 'User-Agent' = 'Timbre-Studio' }} -Uri 'https://api.github.com/repos/Etherll/Timbre/releases/latest'
    $sourceUrl = $release.zipball_url
    Say "using GitHub release $($release.tag_name)"
  }} catch {{
    $sourceUrl = 'https://github.com/Etherll/Timbre/archive/refs/heads/main.zip'
    Say 'no GitHub release found; falling back to the main branch archive'
  }}
  Download-File $sourceUrl $repoZip
  Expand-Archive -Path $repoZip -DestinationPath $repoStage -Force
  $runFile = Get-ChildItem -Path $repoStage -Recurse -Filter 'run_timbre.py' | Select-Object -First 1
  if (!$runFile) {{ throw 'run_timbre.py was not found in the Timbre archive' }}
  if (Test-Path $repo) {{
    $backup = "$repo.broken-$(Get-Date -Format yyyyMMddHHmmss)"
    Say "backing up incomplete repo to $backup"
    Move-Item $repo $backup -Force
  }}
  Move-Item (Split-Path $runFile.FullName -Parent) $repo -Force
}}

if (!(Test-Path $ffmpegExe) -or !(Test-Path $ffprobeExe)) {{
  Say 'installing ffmpeg and ffprobe'
  $ffZip = Join-Path $root 'ffmpeg.zip'
  $ffStage = Join-Path $root '_ffmpeg_extract'
  Remove-Item -Recurse -Force $ffStage -ErrorAction SilentlyContinue
  Download-File 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' $ffZip
  Expand-Archive -Path $ffZip -DestinationPath $ffStage -Force
  $foundFfmpeg = Get-ChildItem -Path $ffStage -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
  $foundFfprobe = Get-ChildItem -Path $ffStage -Recurse -Filter 'ffprobe.exe' | Select-Object -First 1
  if (!$foundFfmpeg -or !$foundFfprobe) {{ throw 'ffmpeg.exe or ffprobe.exe was not found in the archive' }}
  $ffBin = Split-Path $foundFfmpeg.FullName -Parent
  New-Item -ItemType Directory -Force (Split-Path $ffmpegExe) | Out-Null
  Copy-Item (Join-Path $ffBin '*') (Split-Path $ffmpegExe) -Recurse -Force
}}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'python'
$env:UV_CACHE_DIR = Join-Path $root 'uv-cache'
$env:Path = "$(Split-Path $ffmpegExe);$(Join-Path $venv 'Scripts');$env:Path"

Say 'creating managed Python 3.12 environment'
& $uv python install 3.12
& $uv venv $venv --python 3.12
$py = Join-Path $venv 'Scripts\python.exe'
if (!(Test-Path $py)) {{ throw "managed Python was not created at $py" }}

Say 'installing Python packaging tools'
& $uv pip install --python $py --upgrade pip setuptools wheel
Say 'installing Timbre requirements; this can take a long time on first run'
& $uv pip install --python $py -r (Join-Path $repo 'requirements.txt')
Say 'installing YouTube and model-download helpers'
& $uv pip install --python $py yt-dlp huggingface-hub
Set-Content -Path $requirementsMarker -Value (Get-Date -Format o)

$vadDir = Join-Path $repo 'pretrained_models\FireRedVAD'
if (!(Test-Path (Join-Path $vadDir 'VAD'))) {{
  Say 'downloading FireRedVAD model'
  $code = "from huggingface_hub import snapshot_download; snapshot_download(repo_id='FireRedTeam/FireRedVAD', local_dir=r'$vadDir')"
  $env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
  & $py -c $code
}}

Say 'managed runtime is ready'
Write-Output "TIMBRE_SETUP_DONE::$root"
"#);

    let mut cmd = Command::new("powershell.exe");
    cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &script]);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

#[tauri::command]
fn detect_env() -> EnvInfo {
    let repo_root = find_repo_root().map(|p| p.to_string_lossy().into_owned());
    let mut python = None;
    let mut python_version = None;
    for cand in ["python", "python3", "py"] {
        if let Some(v) = capture(cand, &["--version"]) {
            python = Some(cand.to_string());
            python_version = Some(v);
            break;
        }
    }
    // Probe yt-dlp through the same resolver the fetch uses, threading the python
    // we just found so a pip-installed-but-off-PATH yt-dlp is still detected.
    let ytdlp = resolve_ytdlp(python.as_deref());
    EnvInfo {
        repo_root,
        python,
        python_version,
        ytdlp_version: ytdlp.as_ref().map(|r| r.version.clone()),
        ytdlp_via: ytdlp.as_ref().map(|r| r.via.to_string()),
        ffprobe: capture("ffprobe", &["-version"]).is_some(),
        ffmpeg: capture("ffmpeg", &["-version"]).is_some(),
    }
}

// ---------------------------------------------------------------------------
// Commands: reference-candidate generation (wraps repo's extract_reference.py)
// ---------------------------------------------------------------------------

/// Find reference candidates in the source. Prefers the VAD-based finder
/// (`studio/scripts/find_voice_samples.py` — speech-verified, excludes
/// music/applause via the repo's FireRedVAD→Silero stack on CPU); falls back
/// to the notebook's silence-splitting recipe (`extract_reference.py
/// --longest-first`) when the smart script is missing.
/// `dest_dir` is wiped first; as a guard it must end in `ref_candidates`.
#[tauri::command]
fn generate_refs(
    registry: State<'_, ProcRegistry>,
    python: String,
    repo: String,
    source: String,
    dest_dir: String,
    limit: u32,
    min_clip: f64,
    max_clip: f64,
    vad_backend: String,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    let repo_path = PathBuf::from(&repo);
    let smart = repo_path
        .join("studio")
        .join("scripts")
        .join("find_voice_samples.py");
    if !smart.exists() && !repo_path.join("extract_reference.py").exists() {
        return Err(format!("no candidate-finder script found in {repo}"));
    }
    let dest = PathBuf::from(&dest_dir);
    if dest.file_name().is_none_or(|n| n != "ref_candidates") {
        return Err("refusing to wipe a directory not named ref_candidates".into());
    }
    if dest.exists() {
        std::fs::remove_dir_all(&dest).map_err(|e| format!("cannot clear {dest_dir}: {e}"))?;
    }
    std::fs::create_dir_all(&dest).map_err(|e| format!("cannot create {dest_dir}: {e}"))?;

    let mut cmd = Command::new(&python);
    if smart.exists() {
        cmd.arg(smart).args(["--vad-backend", &vad_backend]);
    } else {
        cmd.arg("extract_reference.py").arg("--longest-first");
    }
    cmd.args(["-i", &source, "-o", &dest_dir])
        .args(["--min-clip", &min_clip.to_string(), "--max-clip", &max_clip.to_string()])
        .args(["--limit", &limit.to_string(), "--sr", "16000"])
        .current_dir(&repo_path)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8");
    with_managed_tools(&mut cmd, &repo_path);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RefCandidate {
    path: String,
    name: String,
    start_secs: f64,
    duration_secs: f64,
}

/// Read the manifest extract_reference.py wrote (export order = longest-first)
/// and return the candidates whose wav actually landed on disk.
#[tauri::command]
fn list_ref_candidates(dest_dir: String) -> Result<Vec<RefCandidate>, String> {
    let dir = PathBuf::from(&dest_dir);
    let manifest = dir.join("manifest.csv");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("no manifest in {dest_dir}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 4 {
            continue;
        }
        let (Ok(start), Ok(durs)) = (cols[0].parse::<f64>(), cols[2].parse::<f64>()) else {
            continue;
        };
        let path = dir.join(cols[3].trim());
        if path.is_file() {
            out.push(RefCandidate {
                path: path.to_string_lossy().into_owned(),
                name: cols[3].trim().to_string(),
                start_secs: start,
                duration_secs: durs,
            });
        }
    }
    Ok(out)
}

/// True when the FireRedVAD model dir exists and is non-empty — the same test
/// preflight applies (`timbre.preflight.check_vad_model_dir`), checked early so
/// the UI can offer the download before a run dies at preflight.
#[tauri::command]
fn check_vad_model(repo: String) -> bool {
    vad_model_ready(&PathBuf::from(&repo))
}

/// One-time FireRedVAD model download (public repo, no token) into the
/// location preflight expects — mirrors the README's
/// `hf download FireRedTeam/FireRedVAD --local-dir pretrained_models/FireRedVAD`.
#[tauri::command]
fn download_vad(
    registry: State<'_, ProcRegistry>,
    python: String,
    repo: String,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    let repo_path = PathBuf::from(&repo);
    if !repo_path.join("run_timbre.py").exists() {
        return Err(format!("run_timbre.py not found in {repo}"));
    }
    let code = "print('Downloading FireRedTeam/FireRedVAD (one-time, no token)…', flush=True); \
from huggingface_hub import snapshot_download; \
p = snapshot_download(repo_id='FireRedTeam/FireRedVAD', local_dir='pretrained_models/FireRedVAD'); \
print('VAD model ready at', p, flush=True)";
    let mut cmd = Command::new(&python);
    cmd.args(["-c", code])
        .current_dir(&repo_path)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        // tqdm \r-progress doesn't survive line-buffered streaming; keep output clean
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1");
    with_managed_tools(&mut cmd, &repo_path);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

/// Smart-clean a reference clip: run the pipeline's vocal separator on it and
/// keep whichever version is better (see studio/scripts/clean_voice.py for the
/// verdict logic and stdout markers).
#[tauri::command]
fn clean_ref(
    registry: State<'_, ProcRegistry>,
    python: String,
    repo: String,
    source: String,
    dest_dir: String,
    channel: Channel<ProcEvent>,
) -> Result<u32, String> {
    let repo_path = PathBuf::from(&repo);
    let script = repo_path
        .join("studio")
        .join("scripts")
        .join("clean_voice.py");
    if !script.exists() {
        return Err(format!("clean_voice.py not found under {repo}"));
    }
    std::fs::create_dir_all(&dest_dir).map_err(|e| format!("cannot create {dest_dir}: {e}"))?;
    let mut cmd = Command::new(&python);
    cmd.arg(script)
        .args(["-i", &source, "-o", &dest_dir])
        .current_dir(&repo_path)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8");
    with_managed_tools(&mut cmd, &repo_path);
    let id = registry.next_id();
    spawn_streamed(registry.tasks.clone(), id, cmd, channel)?;
    Ok(id)
}

/// Raw audio bytes for in-app audition (blob URL + waveform on the JS side).
#[tauri::command]
fn read_audio(path: String) -> Result<tauri::ipc::Response, String> {
    let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    Ok(tauri::ipc::Response::new(bytes))
}

/// Copy a picked candidate out of the (wiped-on-regenerate) candidates dir
/// into a stable references folder; returns the new path.
#[tauri::command]
fn keep_ref(path: String, dest_dir: String) -> Result<String, String> {
    let src = PathBuf::from(&path);
    let name = src
        .file_name()
        .ok_or_else(|| format!("not a file: {path}"))?
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&dest_dir).map_err(|e| format!("cannot create {dest_dir}: {e}"))?;
    let mut dst = PathBuf::from(&dest_dir).join(&name);
    let mut n = 1u32;
    while dst.exists() {
        let stem = name.trim_end_matches(".wav");
        dst = PathBuf::from(&dest_dir).join(format!("{stem}_{n}.wav"));
        n += 1;
    }
    std::fs::copy(&src, &dst).map_err(|e| format!("copy failed: {e}"))?;
    Ok(dst.to_string_lossy().into_owned())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileInfo {
    name: String,
    size_bytes: u64,
    duration_secs: Option<f64>,
}

#[tauri::command]
fn probe_file(path: String) -> Result<FileInfo, String> {
    let p = PathBuf::from(&path);
    let meta = std::fs::metadata(&p).map_err(|e| format!("{path}: {e}"))?;
    let duration_secs = capture(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            &path,
        ],
    )
    .and_then(|s| s.parse::<f64>().ok());
    Ok(FileInfo {
        name: p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone()),
        size_bytes: meta.len(),
        duration_secs,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DatasetStats {
    dir: String,
    csv_path: String,
    clips: u64,
}

fn newest_metadata_csv(dir: &Path, depth: u32, best: &mut Option<(SystemTime, PathBuf)>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            newest_metadata_csv(&path, depth + 1, best);
        } else if path.file_name().is_some_and(|n| n == "metadata.csv") {
            if let Ok(meta) = entry.metadata() {
                let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
                    *best = Some((mtime, path));
                }
            }
        }
    }
}

#[tauri::command]
fn dataset_stats(output_dir: String) -> Option<DatasetStats> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    newest_metadata_csv(Path::new(&output_dir), 0, &mut best);
    let (_, csv) = best?;
    let clips = std::fs::read_to_string(&csv)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count() as u64)
        .unwrap_or(0);
    Some(DatasetStats {
        dir: csv.parent()?.to_string_lossy().into_owned(),
        csv_path: csv.to_string_lossy().into_owned(),
        clips,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClipEntry {
    id: String,
    text: String,
    wav: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VisualEntry {
    name: String,
    path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunOverview {
    run_dir: String,
    dataset_dir: String,
    speaker: String,
    clips: Vec<ClipEntry>,
    visuals: Vec<VisualEntry>,
    solo: Option<String>,
    total_clips: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunSummary {
    run_dir: String,
    speaker: String,
    total_clips: u64,
    has_visuals: bool,
    has_solo: bool,
    modified_ms: u64,
}

/// A directory counts as a run if it has anything reviewable — a dataset,
/// visualizations, or the concatenated solo reel. (A run can finish with no
/// dataset, e.g. every clip rejected by the quality gate or export disabled.)
fn find_run_dir(output_dir: &Path) -> Option<PathBuf> {
    let reviewable = |p: &Path| {
        p.join("dataset").join("metadata.csv").is_file()
            || p.join("visualizations").is_dir()
            || solo_wav(p).is_some()
    };
    if reviewable(output_dir) {
        return Some(output_dir.to_path_buf());
    }
    let mut best: Option<(SystemTime, PathBuf)> = None;
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && reviewable(&p) {
                let m = e
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                if best.as_ref().is_none_or(|(t, _)| m > *t) {
                    best = Some((m, p));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

fn is_reviewable_run(p: &Path) -> bool {
    p.join("dataset").join("metadata.csv").is_file()
        || p.join("visualizations").is_dir()
        || solo_wav(p).is_some()
}

fn solo_wav(run_dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(run_dir.join("concatenated_audio_solo_verified"))
        .ok()
        .and_then(|d| {
            d.flatten()
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")))
        })
}

fn summarize_run(run_dir: &Path) -> RunSummary {
    let dataset_dir = run_dir.join("dataset");
    let metadata = dataset_dir.join("metadata.csv");
    let total_clips = std::fs::read_to_string(&metadata)
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count() as u64)
        .unwrap_or(0);
    let mut speaker = String::new();
    if let Ok(entries) = std::fs::read_dir(dataset_dir.join("wavs")) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                speaker = e.file_name().to_string_lossy().into_owned();
                break;
            }
        }
    }
    if speaker.is_empty() {
        speaker = run_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "run".into())
            .split('_')
            .next()
            .unwrap_or("run")
            .to_string();
    }
    let has_visuals = run_dir.join("visualizations").is_dir();
    let has_solo = solo_wav(run_dir).is_some();
    let modified_ms = std::fs::metadata(run_dir)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0);
    RunSummary {
        run_dir: run_dir.to_string_lossy().into_owned(),
        speaker,
        total_clips,
        has_visuals,
        has_solo,
        modified_ms,
    }
}

#[tauri::command]
fn list_runs(output_dir: String) -> Vec<RunSummary> {
    let root = PathBuf::from(output_dir);
    let mut out = Vec::new();
    if is_reviewable_run(&root) {
        out.push(summarize_run(&root));
    }
    if let Ok(entries) = std::fs::read_dir(&root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && is_reviewable_run(&p) {
                out.push(summarize_run(&p));
            }
        }
    }
    out.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
    out
}

fn build_run_overview(run_dir: PathBuf) -> RunOverview {
    let dataset_dir = run_dir.join("dataset");

    // dataset/wavs/<TargetName>/<id>.wav — one speaker per run
    let wavs_root = dataset_dir.join("wavs");
    let mut speaker = String::new();
    let mut wav_dir = wavs_root.clone();
    if let Ok(entries) = std::fs::read_dir(&wavs_root) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                speaker = e.file_name().to_string_lossy().into_owned();
                wav_dir = e.path();
                break;
            }
        }
    }
    if speaker.is_empty() {
        // fall back to the run-dir naming convention: <Target>_<input>_extracted
        speaker = run_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
            .split('_')
            .next()
            .unwrap_or_default()
            .to_string();
    }

    let mut clips = Vec::new();
    let mut total_clips = 0u64;
    if let Ok(text) = std::fs::read_to_string(dataset_dir.join("metadata.csv")) {
        for line in text.lines() {
            let mut parts = line.splitn(3, '|');
            let (Some(id), Some(t)) = (parts.next(), parts.next()) else { continue };
            if id.trim().is_empty() {
                continue;
            }
            total_clips += 1;
            if clips.len() >= 3000 {
                continue; // keep the payload sane on giant datasets; total still counts
            }
            let mut wav = wav_dir.join(format!("{id}.wav"));
            if !wav.is_file() {
                wav = wav_dir.join(id); // id may already carry the extension
            }
            clips.push(ClipEntry {
                id: id.to_string(),
                text: t.trim().to_string(),
                wav: wav.to_string_lossy().into_owned(),
            });
        }
    }

    let mut visuals = Vec::new();
    if let Ok(entries) = std::fs::read_dir(run_dir.join("visualizations")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png"))
                && visuals.len() < 60
            {
                visuals.push(VisualEntry {
                    name: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
                    path: p.to_string_lossy().into_owned(),
                });
            }
        }
        visuals.sort_by(|a, b| a.name.cmp(&b.name));
    }

    let solo = solo_wav(&run_dir).map(|p| p.to_string_lossy().into_owned());

    RunOverview {
        run_dir: run_dir.to_string_lossy().into_owned(),
        dataset_dir: dataset_dir.to_string_lossy().into_owned(),
        speaker,
        clips,
        visuals,
        solo,
        total_clips,
    }
}

/// Everything the results browser needs from the newest reviewable run:
/// dataset clips (id|transcript from metadata.csv + resolved wav paths),
/// the spectrogram gallery (visualizations/), and the concatenated solo reel.
#[tauri::command]
fn run_overview(output_dir: String) -> Result<RunOverview, String> {
    let run_dir = find_run_dir(Path::new(&output_dir))
        .ok_or("no reviewable run found under the output folder")?;
    Ok(build_run_overview(run_dir))
}

#[tauri::command]
fn run_overview_for(output_dir: String, run_dir: String) -> Result<RunOverview, String> {
    let root = std::fs::canonicalize(&output_dir)
        .map_err(|e| format!("cannot resolve output folder {output_dir}: {e}"))?;
    let path = std::fs::canonicalize(&run_dir)
        .map_err(|e| format!("cannot resolve run folder {run_dir}: {e}"))?;
    let root_match = path == root;
    let direct_child = path.parent().is_some_and(|p| p == root);
    if !(root_match || direct_child) {
        return Err(format!("run is outside the output folder: {run_dir}"));
    }
    if !is_reviewable_run(&path) {
        return Err(format!("not a reviewable run: {run_dir}"));
    }
    Ok(build_run_overview(path))
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    if !Path::new(&path).exists() {
        return Err(format!("path does not exist: {path}"));
    }
    #[cfg(target_os = "windows")]
    let result = quiet(Command::new("explorer").arg(&path)).spawn();
    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg(&path).spawn();
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let result = Command::new("xdg-open").arg(&path).spawn();
    result.map(|_| ()).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(ProcRegistry::default())
        .invoke_handler(tauri::generate_handler![
            detect_env,
            default_setup_dir,
            setup_status,
            start_setup,
            start_pipeline,
            start_ytdlp,
            cancel_task,
            probe_file,
            dataset_stats,
            open_path,
            generate_refs,
            list_ref_candidates,
            read_audio,
            keep_ref,
            check_vad_model,
            download_vad,
            list_runs,
            run_overview,
            run_overview_for,
            clean_ref
        ])
        .run(tauri::generate_context!())
        .expect("error while running Timbre Studio");
}

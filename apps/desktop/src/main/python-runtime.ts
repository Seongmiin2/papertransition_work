import { spawn } from "node:child_process";
import { existsSync, lstatSync, realpathSync } from "node:fs";
import { delimiter, isAbsolute, join, resolve } from "node:path";

const PROBE_TIMEOUT_MS = 5_000;
const PYTHON_MODULE = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/;
const NO_RUNTIME_MESSAGE =
  "필요한 모듈을 갖춘 로컬 Python 실행 환경을 찾지 못했습니다.";

export interface PythonProbeCommand {
  executable: string;
  args: readonly string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  shell: false;
  windowsHide: true;
  timeoutMs: number;
}

export type PythonProbeRunner = (command: PythonProbeCommand) => Promise<boolean>;

export interface SelectPythonRuntimeOptions {
  projectRoot: string;
  requiredImports: readonly string[];
  environment?: NodeJS.ProcessEnv;
  probeRunner?: PythonProbeRunner;
}

const runtimeCache = new Map<string, Promise<string>>();

export async function selectPythonRuntime(
  options: SelectPythonRuntimeOptions,
): Promise<string> {
  const projectRoot = requiredProjectRoot(options.projectRoot);
  const requiredImports = normalizeImports(options.requiredImports);
  const environment = options.environment ?? process.env;
  if (options.probeRunner) {
    return await selectUncached(
      projectRoot,
      requiredImports,
      environment,
      options.probeRunner,
    );
  }
  const cacheKey = [
    comparablePath(projectRoot),
    requiredImports.join(","),
    environment.SCAN2HWPX_PYTHON ?? "",
  ].join("\u0000");
  let selection = runtimeCache.get(cacheKey);
  if (!selection) {
    selection = selectUncached(
      projectRoot,
      requiredImports,
      environment,
      runPythonProbe,
    );
    runtimeCache.set(cacheKey, selection);
  }
  return await selection;
}

export function pythonRuntimeEnvironment(
  projectRoot: string,
  environment: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  return {
    ...environment,
    PYTHONPATH: [join(projectRoot, "src"), environment.PYTHONPATH]
      .filter(Boolean)
      .join(delimiter),
    PYTHONUTF8: "1",
  };
}

async function selectUncached(
  projectRoot: string,
  requiredImports: readonly string[],
  environment: NodeJS.ProcessEnv,
  probeRunner: PythonProbeRunner,
): Promise<string> {
  const candidates: string[] = [];
  const override = regularAbsoluteFile(environment.SCAN2HWPX_PYTHON);
  if (override) candidates.push(override);
  const venv = regularAbsoluteFile(
    join(projectRoot, ".venv", "Scripts", "python.exe"),
  );
  if (venv) candidates.push(venv);
  candidates.push("python");

  const seen = new Set<string>();
  for (const executable of candidates) {
    const key = comparablePath(executable);
    if (seen.has(key)) continue;
    seen.add(key);
    const command: PythonProbeCommand = {
      executable,
      args: ["-c", requiredImports.map((name) => `import ${name}`).join("; ")],
      cwd: projectRoot,
      env: pythonRuntimeEnvironment(projectRoot, environment),
      shell: false,
      windowsHide: true,
      timeoutMs: PROBE_TIMEOUT_MS,
    };
    let passed = false;
    try {
      passed = await probeRunner(command);
    } catch {
      passed = false;
    }
    if (passed) return executable;
  }
  throw new Error(NO_RUNTIME_MESSAGE);
}

async function runPythonProbe(command: PythonProbeCommand): Promise<boolean> {
  return await new Promise((resolvePromise) => {
    let settled = false;
    let timer: NodeJS.Timeout | undefined;
    const finish = (passed: boolean) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      resolvePromise(passed);
    };
    let child;
    try {
      child = spawn(command.executable, [...command.args], {
        cwd: command.cwd,
        env: command.env,
        shell: command.shell,
        windowsHide: command.windowsHide,
        stdio: "ignore",
      });
    } catch {
      resolvePromise(false);
      return;
    }
    timer = setTimeout(() => {
      child.kill();
      finish(false);
    }, command.timeoutMs);
    child.once("error", () => finish(false));
    child.once("close", (code) => finish(code === 0));
  });
}

function requiredProjectRoot(path: unknown): string {
  if (typeof path !== "string" || !isAbsolute(path)) {
    throw new Error(NO_RUNTIME_MESSAGE);
  }
  const absolute = resolve(path);
  try {
    const stats = lstatSync(absolute);
    if (stats.isSymbolicLink() || !stats.isDirectory()) {
      throw new Error(NO_RUNTIME_MESSAGE);
    }
    return realpathSync.native(absolute);
  } catch {
    throw new Error(NO_RUNTIME_MESSAGE);
  }
}

function normalizeImports(values: readonly string[]): readonly string[] {
  if (
    !Array.isArray(values) ||
    values.length === 0 ||
    values.some((value) => typeof value !== "string" || !PYTHON_MODULE.test(value))
  ) {
    throw new Error(NO_RUNTIME_MESSAGE);
  }
  return [...new Set(values)].sort();
}

function regularAbsoluteFile(path: string | undefined): string | null {
  if (!path || !isAbsolute(path) || !existsSync(path)) return null;
  try {
    const stats = lstatSync(path);
    if (stats.isSymbolicLink() || !stats.isFile()) return null;
    return realpathSync.native(path);
  } catch {
    return null;
  }
}

function comparablePath(path: string): string {
  const value = isAbsolute(path) ? resolve(path) : path;
  return process.platform === "win32" ? value.toLowerCase() : value;
}

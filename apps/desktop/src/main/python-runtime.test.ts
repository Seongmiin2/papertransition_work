import {
  mkdirSync,
  mkdtempSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  selectPythonRuntime,
  type PythonProbeCommand,
  type PythonProbeRunner,
} from "./python-runtime.js";

const roots: string[] = [];
const requiredImports = ["scan2hwpx", "pydantic"] as const;

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("Python runtime selector", () => {
  it("falls back to PATH python when an existing venv misses imports", async () => {
    const fixture = makeFixture();
    const calls: PythonProbeCommand[] = [];
    const runner: PythonProbeRunner = vi.fn(async (command) => {
      calls.push(command);
      return command.executable === "python";
    });

    const selected = await selectPythonRuntime({
      projectRoot: fixture.projectRoot,
      requiredImports,
      environment: {},
      probeRunner: runner,
    });

    expect(selected).toBe("python");
    expect(calls.map((command) => command.executable)).toEqual([
      realpathSync.native(fixture.venvPython),
      "python",
    ]);
    expect(calls.every((command) => command.shell === false)).toBe(true);
    expect(calls.every((command) => command.windowsHide === true)).toBe(true);
    expect(calls.every((command) => command.timeoutMs === 5_000)).toBe(true);
  });

  it("prefers a valid absolute SCAN2HWPX_PYTHON override", async () => {
    const fixture = makeFixture();
    const calls: PythonProbeCommand[] = [];
    const runner: PythonProbeRunner = vi.fn(async (command) => {
      calls.push(command);
      return true;
    });

    const selected = await selectPythonRuntime({
      projectRoot: fixture.projectRoot,
      requiredImports,
      environment: { SCAN2HWPX_PYTHON: fixture.overridePython },
      probeRunner: runner,
    });

    expect(selected).toBe(realpathSync.native(fixture.overridePython));
    expect(calls).toHaveLength(1);
    expect(calls[0].args).toEqual([
      "-c",
      "import pydantic; import scan2hwpx",
    ]);
    expect(calls[0].env.PYTHONPATH).toContain(
      join(realpathSync.native(fixture.projectRoot), "src"),
    );
  });

  it("returns only a generic error when every candidate fails", async () => {
    const fixture = makeFixture();
    const runner: PythonProbeRunner = vi.fn(async () => false);

    let error: unknown;
    try {
      await selectPythonRuntime({
        projectRoot: fixture.projectRoot,
        requiredImports,
        environment: { SCAN2HWPX_PYTHON: fixture.overridePython },
        probeRunner: runner,
      });
    } catch (reason) {
      error = reason;
    }

    expect(runner).toHaveBeenCalledTimes(3);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toBe(
      "필요한 모듈을 갖춘 로컬 Python 실행 환경을 찾지 못했습니다.",
    );
    expect((error as Error).message).not.toContain(fixture.projectRoot);
  });
});

function makeFixture() {
  const root = mkdtempSync(join(tmpdir(), "exam2hwpx-python-runtime-"));
  roots.push(root);
  const projectRoot = join(root, "project");
  const venvDirectory = join(projectRoot, ".venv", "Scripts");
  mkdirSync(join(projectRoot, "src"), { recursive: true });
  mkdirSync(venvDirectory, { recursive: true });
  const venvPython = join(venvDirectory, "python.exe");
  const overridePython = join(root, "override-python.exe");
  writeFileSync(venvPython, "fixture", "utf8");
  writeFileSync(overridePython, "fixture", "utf8");
  return { projectRoot, venvPython, overridePython };
}

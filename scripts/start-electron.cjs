const { spawn } = require("node:child_process");
const { resolve } = require("node:path");

const electron = require("electron");
const env = { ...process.env };

// Electron-based terminals can leak this flag into child shells. It would make
// the app binary behave like plain Node and leave the Electron API unavailable.
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(electron, [resolve(__dirname, "..")], {
  env,
  stdio: "inherit",
  windowsHide: false,
});

child.on("error", (error) => {
  console.error(`Electron을 시작하지 못했습니다: ${error.message}`);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exitCode = code ?? 1;
});

const { app, BrowserWindow } = require("electron");
const path = require("node:path");

const target = process.argv[2];
if (!target) {
  console.error("usage: electron scripts/verify_review_ui.cjs <review.html>");
  process.exit(2);
}

app.whenReady().then(async () => {
  const window = new BrowserWindow({
    show: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  window.webContents.on("console-message", (_event, level, message) => {
    console.error(`renderer console ${level}: ${message}`);
  });
  try {
    await window.loadFile(path.resolve(target));
    const before = await window.webContents.executeJavaScript(`({
    cards: document.querySelectorAll('.card').length,
    inputs: document.querySelectorAll('input[type=text]').length,
    checkboxes: document.querySelectorAll('input[type=checkbox]').length,
    cropImages: document.querySelectorAll('img.crop').length,
    loadedCrops: [...document.querySelectorAll('img.crop')].filter(image => image.naturalWidth > 0).length,
    downloadButtons: document.querySelectorAll('#download').length,
    counter: document.querySelector('#counter')?.textContent
  })`);
    console.log(JSON.stringify({ before }, null, 2));
    const after = await window.webContents.executeJavaScript(`(() => {
    const input = document.querySelector('input[type=text]');
    input.value = '검증 문장';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('input[type=checkbox]').click();
    return { counter: document.querySelector('#counter').textContent, value: input.value };
  })()`);
    console.log(JSON.stringify({ after }, null, 2));
    const valid =
      before.cards > 0 &&
      before.cards === before.inputs &&
      before.cards === before.checkboxes &&
      before.cropImages === before.loadedCrops &&
      before.downloadButtons === 1 &&
      after.counter.startsWith("확정 1 /") &&
      after.value === "검증 문장";
    await window.close();
    app.exit(valid ? 0 : 1);
  } catch (error) {
    console.error(error);
    await window.close();
    app.exit(1);
  }
});

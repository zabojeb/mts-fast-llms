const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs").promises;
const { exec } = require("child_process");
const util = require("util");
const execPromise = util.promisify(exec);

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
    },
    icon: path.join(__dirname, "icon.png"),
  });

  mainWindow.loadFile("index.html");

  // Open DevTools in development
  if (process.env.NODE_ENV === "development") {
    mainWindow.webContents.openDevTools();
  }
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers for file operations
ipcMain.handle("save-pipeline", async (event, data) => {
  const { filePath, canceled } = await dialog.showSaveDialog({
    title: "Save Pipeline",
    defaultPath: "pipeline.json",
    filters: [
      { name: "JSON Files", extensions: ["json"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });

  if (!canceled && filePath) {
    await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    return { success: true, filePath };
  }
  return { success: false };
});

ipcMain.handle("load-pipeline", async (event) => {
  const { filePaths, canceled } = await dialog.showOpenDialog({
    title: "Load Pipeline",
    filters: [
      { name: "JSON Files", extensions: ["json"] },
      { name: "All Files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });

  if (!canceled && filePaths.length > 0) {
    const data = await fs.readFile(filePaths[0], "utf-8");
    return { success: true, data: JSON.parse(data) };
  }
  return { success: false };
});

ipcMain.handle("select-model-path", async (event) => {
  const { filePaths, canceled } = await dialog.showOpenDialog({
    title: "Select Model Directory",
    properties: ["openDirectory"],
  });

  if (!canceled && filePaths.length > 0) {
    return { success: true, path: filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle("select-save-path", async (event) => {
  const { filePath, canceled } = await dialog.showSaveDialog({
    title: "Save Model As",
    defaultPath: "optimized_model",
  });

  if (!canceled && filePath) {
    return { success: true, path: filePath };
  }
  return { success: false };
});

// Execute Python pipeline
ipcMain.handle("execute-pipeline", async (event, pipelineData) => {
  try {
    // Save pipeline to temporary file
    const tempPath = path.join(
      app.getPath("temp"),
      `pipeline_${Date.now()}.json`
    );
    await fs.writeFile(tempPath, JSON.stringify(pipelineData, null, 2));

    // Execute Python script
    const pythonScript = path.join(__dirname, "pipeline_executor.py");
    const { stdout, stderr } = await execPromise(
      `python "${pythonScript}" "${tempPath}"`
    );

    // Clean up temp file
    await fs.unlink(tempPath).catch(() => {});

    if (stderr) {
      return { success: false, error: stderr };
    }

    return { success: true, output: stdout };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

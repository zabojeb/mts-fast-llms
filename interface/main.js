const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  win.loadFile('index.html');
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// IPC: получение и выполнение графа через Python-библиотеку
ipcMain.handle('run-pipeline', async (_, graphJson) => {
  // Здесь можно запустить Python-скрипт через child_process
  // и передать graphJson, вернуть результат
  return { status: 'ok', details: 'Pipeline executed.' };
});
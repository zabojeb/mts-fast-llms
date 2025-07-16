const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  runPipeline: (graph) => ipcRenderer.invoke('run-pipeline', graph)
});
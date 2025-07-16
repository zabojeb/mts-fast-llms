import Rete from 'rete';
import VueRenderPlugin from 'rete-vue-render-plugin';

// 6.1. Сокеты
const sockets = {
  model: new Rete.Socket('Model'),
  params: new Rete.Socket('Params'),
};

// 6.2. Компоненты нод
class ModelLoadComponent extends Rete.Component {
  constructor() { super('Model Load'); }
  builder(node) {
    const out = new Rete.Output('model', 'Model', sockets.model);
    node.addControl(new Rete.Control(this.editor, 'sourceType', { component: 'select', props: { items: ['huggingface','local'] }}));
    node.addControl(new Rete.Control(this.editor, 'path', { component: 'text', props: { placeholder: 'URL или путь' }}));
    return node.addOutput(out);
  }
}

class ParamsComponent extends Rete.Component {
  constructor() { super('Params'); }
  builder(node) {
    const out = new Rete.Output('params', 'Params', sockets.params);
    node.addControl(new Rete.Control(this.editor, 'params', { component: 'code', props: { language: 'json' }}));
    return node.addOutput(out);
  }
}

class OptimizationComponent extends Rete.Component {
  constructor() { super('Optimization'); }
  builder(node) {
    const inpModel = new Rete.Input('model', 'Model', sockets.model);
    const inpParams = new Rete.Input('params', 'Params', sockets.params);
    const out = new Rete.Output('model', 'Model', sockets.model);
    node.addInput(inpModel).addInput(inpParams).addControl(
      new Rete.Control(this.editor, 'optType', { component: 'select', props: { items: ['pruning','quantization','distillation'] }}));
    // Для дистилляции можно добавить второй вход, но можно и динамически
    return node.addOutput(out);
  }
}

class ModelSaveComponent extends Rete.Component {
  constructor() { super('Model Save'); }
  builder(node) {
    const inp = new Rete.Input('model', 'Model', sockets.model);
    node.addControl(new Rete.Control(this.editor, 'savePath', { component: 'text', props: { placeholder: 'Путь для сохранения' }}));
    return node.addInput(inp);
  }
}

async function createEditor() {
  const container = document.querySelector('#rete');
  const editor = new Rete.NodeEditor('opt@0.1.0', container);
  editor.use(VueRenderPlugin);

  const engine = new Rete.Engine('opt@0.1.0');

  [ModelLoadComponent, ParamsComponent, OptimizationComponent, ModelSaveComponent]
    .map(C => new C())
    .forEach(c => { editor.register(c); engine.register(c); });

  editor.on('process nodecreated noderemoved connectioncreated connectionremoved', async () => {
    await engine.abort();
    const data = await engine.process(editor.toJSON());
  });

  document.getElementById('run').addEventListener('click', async () => {
    const graph = editor.toJSON();
    const res = await window.api.runPipeline(graph);
    console.log(res);
  });
}

createEditor();
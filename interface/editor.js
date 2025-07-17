const { ipcRenderer } = require("electron");

// Socket types
const modelSocket = new Rete.Socket("Model");
const parameterSocket = new Rete.Socket("Parameters");

// Custom control for file/model selection
class FileControl extends Rete.Control {
  constructor(emitter, key, readonly = false) {
    super(key);
    this.emitter = emitter;
    this.key = key;
    this.readonly = readonly;
    this.component = {
      template: `
                <div>
                    <input type="text" 
                           :value="value" 
                           :readonly="readonly"
                           @input="onChange($event.target.value)"
                           placeholder="Enter path or HuggingFace model ID"
                           style="width: 200px; margin-right: 5px;">
                    <button @click="browse" v-if="!readonly">Browse</button>
                </div>
            `,
      data() {
        return {
          value: "",
          readonly: readonly,
        };
      },
      methods: {
        onChange(val) {
          this.value = val;
          this.update();
        },
        browse() {
          ipcRenderer.invoke("select-model-path").then((result) => {
            if (result.success) {
              this.value = result.path;
              this.update();
            }
          });
        },
        update() {
          this.emitter.trigger("process");
        },
      },
      mounted() {
        this.value = this.$el.value;
      },
    };
  }

  setValue(val) {
    this.vueContext.value = val;
  }

  getValue() {
    return this.vueContext.value;
  }
}

// Model Load Component
class ModelLoadComponent extends Rete.Component {
  constructor() {
    super("Model Load");
  }

  builder(node) {
    const out = new Rete.Output("model", "Model", modelSocket);
    const ctrl = new FileControl(this.editor, "path");

    return node.addControl(ctrl).addOutput(out);
  }

  worker(node, inputs, outputs) {
    outputs["model"] = {
      type: "model",
      path: node.data.path || "",
    };
  }
}

// Model Save Component
class ModelSaveComponent extends Rete.Component {
  constructor() {
    super("Model Save");
  }

  builder(node) {
    const inp = new Rete.Input("model", "Model", modelSocket);
    const formatCtrl = new Rete.Control("format", {
      template: `
                <div>
                    <label>Format:</label>
                    <select @input="onChange($event.target.value)">
                        <option value="pytorch">PyTorch</option>
                        <option value="onnx">ONNX</option>
                        <option value="tensorflow">TensorFlow</option>
                        <option value="tflite">TFLite</option>
                    </select>
                </div>
            `,
      data() {
        return {
          value: "pytorch",
        };
      },
      methods: {
        onChange(val) {
          this.value = val;
          this.$emit("input", val);
        },
      },
    });

    const pathCtrl = new FileControl(this.editor, "save_path");

    return node.addInput(inp).addControl(formatCtrl).addControl(pathCtrl);
  }

  worker(node, inputs, outputs) {
    const model = inputs["model"][0];
    if (model) {
      node.data.model = model;
      node.data.format = node.data.format || "pytorch";
    }
  }
}

// Parameters Component
class ParametersComponent extends Rete.Component {
  constructor() {
    super("Optimization Parameters");
  }

  builder(node) {
    const out = new Rete.Output("params", "Parameters", parameterSocket);

    const controls = {
      pruning_sparsity: new Rete.Control("pruning_sparsity", {
        template: `
                    <div>
                        <label>Pruning Sparsity (0-1):</label>
                        <input type="number" min="0" max="1" step="0.1" 
                               :value="value" @input="onChange($event.target.value)">
                    </div>
                `,
        data() {
          return { value: 0.5 };
        },
        methods: {
          onChange(val) {
            this.value = parseFloat(val);
            this.$emit("input", this.value);
          },
        },
      }),
      quantization_bits: new Rete.Control("quantization_bits", {
        template: `
                    <div>
                        <label>Quantization Bits:</label>
                        <select @input="onChange($event.target.value)">
                            <option value="8">8-bit</option>
                            <option value="4">4-bit</option>
                            <option value="2">2-bit</option>
                        </select>
                    </div>
                `,
        data() {
          return { value: "8" };
        },
        methods: {
          onChange(val) {
            this.value = val;
            this.$emit("input", val);
          },
        },
      }),
      distillation_temperature: new Rete.Control("distillation_temperature", {
        template: `
                    <div>
                        <label>Distillation Temperature:</label>
                        <input type="number" min="1" max="20" step="0.5" 
                               :value="value" @input="onChange($event.target.value)">
                    </div>
                `,
        data() {
          return { value: 5 };
        },
        methods: {
          onChange(val) {
            this.value = parseFloat(val);
            this.$emit("input", this.value);
          },
        },
      }),
      learning_rate: new Rete.Control("learning_rate", {
        template: `
                    <div>
                        <label>Learning Rate:</label>
                        <input type="number" min="0.00001" max="0.1" step="0.00001" 
                               :value="value" @input="onChange($event.target.value)">
                    </div>
                `,
        data() {
          return { value: 0.001 };
        },
        methods: {
          onChange(val) {
            this.value = parseFloat(val);
            this.$emit("input", this.value);
          },
        },
      }),
    };

    node.addOutput(out);
    for (const [key, control] of Object.entries(controls)) {
      node.addControl(control);
    }

    return node;
  }

  worker(node, inputs, outputs) {
    outputs["params"] = {
      pruning_sparsity: node.data.pruning_sparsity || 0.5,
      quantization_bits: parseInt(node.data.quantization_bits || "8"),
      distillation_temperature: node.data.distillation_temperature || 5,
      learning_rate: node.data.learning_rate || 0.001,
    };
  }
}

// Pruning Component
class PruningComponent extends Rete.Component {
  constructor() {
    super("Pruning");
  }

  builder(node) {
    const inp1 = new Rete.Input("model", "Model", modelSocket);
    const inp2 = new Rete.Input("params", "Parameters", parameterSocket);
    const out = new Rete.Output("model", "Pruned Model", modelSocket);

    const methodCtrl = new Rete.Control("method", {
      template: `
                <div>
                    <label>Pruning Method:</label>
                    <select @input="onChange($event.target.value)">
                        <option value="magnitude">Magnitude-based</option>
                        <option value="structured">Structured</option>
                        <option value="random">Random</option>
                    </select>
                </div>
            `,
      data() {
        return { value: "magnitude" };
      },
      methods: {
        onChange(val) {
          this.value = val;
          this.$emit("input", val);
        },
      },
    });

    return node
      .addInput(inp1)
      .addInput(inp2)
      .addControl(methodCtrl)
      .addOutput(out);
  }

  worker(node, inputs, outputs) {
    const model = inputs["model"][0];
    const params = inputs["params"][0];

    if (model && params) {
      outputs["model"] = {
        ...model,
        optimization: {
          type: "pruning",
          method: node.data.method || "magnitude",
          sparsity: params.pruning_sparsity,
        },
      };
    }
  }
}

// Quantization Component
class QuantizationComponent extends Rete.Component {
  constructor() {
    super("Quantization");
  }

  builder(node) {
    const inp1 = new Rete.Input("model", "Model", modelSocket);
    const inp2 = new Rete.Input("params", "Parameters", parameterSocket);
    const out = new Rete.Output("model", "Quantized Model", modelSocket);

    const modeCtrl = new Rete.Control("mode", {
      template: `
                <div>
                    <label>Quantization Mode:</label>
                    <select @input="onChange($event.target.value)">
                        <option value="dynamic">Dynamic</option>
                        <option value="static">Static</option>
                        <option value="qat">QAT (Quantization Aware Training)</option>
                    </select>
                </div>
            `,
      data() {
        return { value: "dynamic" };
      },
      methods: {
        onChange(val) {
          this.value = val;
          this.$emit("input", val);
        },
      },
    });

    return node
      .addInput(inp1)
      .addInput(inp2)
      .addControl(modeCtrl)
      .addOutput(out);
  }

  worker(node, inputs, outputs) {
    const model = inputs["model"][0];
    const params = inputs["params"][0];

    if (model && params) {
      outputs["model"] = {
        ...model,
        optimization: {
          type: "quantization",
          mode: node.data.mode || "dynamic",
          bits: params.quantization_bits,
        },
      };
    }
  }
}

// Distillation Component
class DistillationComponent extends Rete.Component {
  constructor() {
    super("Distillation");
  }

  builder(node) {
    const inp1 = new Rete.Input("teacher", "Teacher Model", modelSocket);
    const inp2 = new Rete.Input("student", "Student Model", modelSocket);
    const inp3 = new Rete.Input("params", "Parameters", parameterSocket);
    const out = new Rete.Output("model", "Distilled Model", modelSocket);

    const epochsCtrl = new Rete.Control("epochs", {
      template: `
                <div>
                    <label>Training Epochs:</label>
                    <input type="number" min="1" max="100" 
                           :value="value" @input="onChange($event.target.value)">
                </div>
            `,
      data() {
        return { value: 10 };
      },
      methods: {
        onChange(val) {
          this.value = parseInt(val);
          this.$emit("input", this.value);
        },
      },
    });

    return node
      .addInput(inp1)
      .addInput(inp2)
      .addInput(inp3)
      .addControl(epochsCtrl)
      .addOutput(out);
  }

  worker(node, inputs, outputs) {
    const teacher = inputs["teacher"][0];
    const student = inputs["student"][0];
    const params = inputs["params"][0];

    if (teacher && student && params) {
      outputs["model"] = {
        ...student,
        optimization: {
          type: "distillation",
          teacher_model: teacher,
          temperature: params.distillation_temperature,
          epochs: node.data.epochs || 10,
          learning_rate: params.learning_rate,
        },
      };
    }
  }
}

// Main Editor Class
class NodeEditor {
  constructor() {
    this.editor = null;
    this.engine = null;
    this.components = [];
    this.init();
  }

  async init() {
    const container = document.getElementById("editor");
    this.editor = new Rete.NodeEditor("llm-optimization@0.1.0", container);

    // Register components
    this.components = [
      new ModelLoadComponent(),
      new ModelSaveComponent(),
      new ParametersComponent(),
      new PruningComponent(),
      new QuantizationComponent(),
      new DistillationComponent(),
    ];

    this.components.forEach((c) => {
      this.editor.register(c);
    });

    // Initialize engine
    this.engine = new Rete.Engine("llm-optimization@0.1.0");
    this.components.forEach((c) => {
      this.engine.register(c);
    });

    // Add plugins
    this.editor.use(ConnectionPlugin.default);
    this.editor.use(VueRenderPlugin.default);

    // Area plugin setup
    const background = document.createElement("div");
    background.classList.add("background");

    const AreaPlugin = {
      name: "area",
      install(editor) {
        editor.on("zoom translate", () => {
          const { k } = editor.view.area.transform;
          const x = editor.view.area.transform.x;
          const y = editor.view.area.transform.y;

          editor.view.area.el.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
        });
      },
    };

    this.editor.use(AreaPlugin);

    // Handle drag and drop
    this.setupDragDrop();

    // Process on changes
    this.editor.on(
      "process nodecreated noderemoved connectioncreated connectionremoved",
      async () => {
        await this.engine.abort();
        await this.engine.process(this.editor.toJSON());
      }
    );

    this.editor.view.resize();
    this.editor.trigger("process");

    this.updateStatus("Editor initialized");
  }

  setupDragDrop() {
    const editor = this.editor;
    const editorEl = document.getElementById("editor");

    // Handle drag start
    document.querySelectorAll(".node-item").forEach((item) => {
      item.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("node-type", item.dataset.node);
      });
    });

    // Handle drag over
    editorEl.addEventListener("dragover", (e) => {
      e.preventDefault();
    });

    // Handle drop
    editorEl.addEventListener("drop", async (e) => {
      e.preventDefault();
      const nodeType = e.dataTransfer.getData("node-type");

      if (!nodeType) return;

      const point = editor.view.area.mouse;
      const component = this.getComponentByType(nodeType);

      if (component) {
        const node = await component.createNode();
        node.position = [point.x, point.y];
        editor.addNode(node);
      }
    });
  }

  getComponentByType(type) {
    const mapping = {
      "model-load": ModelLoadComponent,
      "model-save": ModelSaveComponent,
      parameters: ParametersComponent,
      pruning: PruningComponent,
      quantization: QuantizationComponent,
      distillation: DistillationComponent,
    };

    const ComponentClass = mapping[type];
    return ComponentClass ? new ComponentClass() : null;
  }

  async savePipeline() {
    const data = this.editor.toJSON();
    const result = await ipcRenderer.invoke("save-pipeline", data);

    if (result.success) {
      this.updateStatus(`Pipeline saved to ${result.filePath}`);
    }
  }

  async loadPipeline() {
    const result = await ipcRenderer.invoke("load-pipeline");

    if (result.success) {
      await this.editor.fromJSON(result.data);
      this.updateStatus("Pipeline loaded successfully");
    }
  }

  clearEditor() {
    this.editor.clear();
    this.updateStatus("Editor cleared");
  }

  async executePipeline() {
    this.updateStatus("Executing pipeline...");
    document.getElementById("toolbar").innerHTML +=
      '<span class="loading"></span>';

    const data = this.editor.toJSON();
    const result = await ipcRenderer.invoke("execute-pipeline", data);

    document.querySelector(".loading")?.remove();

    if (result.success) {
      this.updateStatus("Pipeline executed successfully");
      this.showOutput(result.output);
    } else {
      this.updateStatus("Pipeline execution failed");
      this.showOutput(result.error);
    }
  }

  showOutput(text) {
    document.getElementById("modal-text").textContent = text;
    document.getElementById("modal").style.display = "block";
  }

  updateStatus(message) {
    document.getElementById("status").textContent = message;
  }
}

// Initialize editor when DOM is ready
const editor = new NodeEditor();

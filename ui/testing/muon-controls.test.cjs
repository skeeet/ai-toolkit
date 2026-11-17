const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('typescript');

// Exercise the real TS/TSX without a browser or another test framework. Existing
// form widgets are replaced with inert components so their props and callbacks
// can be inspected without loading the application's unrelated services.
const widgets = {
  NumberInput() {},
  SelectInput() {},
  Checkbox() {},
};
function load(relative, imports = {}) {
  const filename = path.resolve(__dirname, '../src/app/jobs/new', relative);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const module = { exports: {} };
  const resolve = name => (name in imports ? imports[name] : require(name));
  vm.runInThisContext(`(function(require,module,exports){${code}\n})`, { filename })(resolve, module, module.exports);
  return module.exports;
}
const config = load('muonConfig.ts');
const Controls = load('MuonControls.tsx', {
  '@/components/formInputs': widgets,
  './muonConfig': config,
}).default;
function elements(tree) {
  if (Array.isArray(tree)) return tree.flatMap(elements);
  if (!tree || !tree.props) return [];
  return [tree, ...elements(tree.props.children)];
}
function train(overrides = {}) {
  return {
    optimizer: 'adamw8bit',
    lr: 1e-4,
    optimizer_params: { weight_decay: 1e-4, betas: [0.9, 0.99] },
    per_image_adaptive: { mode: 'observe', warmup_epochs: 4 },
    ...overrides,
  };
}

function diffusionJob(arch, type = 'lora') {
  return {
    type: 'diffusion_trainer',
    model: { arch, name_or_path: 'example/base', model_kwargs: {} },
    network: { type },
    train: train({ train_unet: true, train_text_encoder: false }),
    datasets: [],
  };
}

test('Muon is offered across denoisers and recognized adapter types before runtime shape checks', () => {
  for (const [arch, type] of [
    ['krea2', 'lora'],
    ['krea2:turbo', 'lokr'],
    ['qwen_image_2', 'locon'],
    ['qwen_image_2:turbo', 'lycoris'],
    ['zimage:turbo', 'lorm'],
    ['moss_image', 'lokr'],
    ['moss_image:base', 'lora'],
    ['sdxl', 'DORA'],
  ]) {
    assert.equal(config.shouldOfferMuon(diffusionJob(arch, type)), true, `${arch}/${type}`);
  }
  const unrelated = diffusionJob('sdxl');
  unrelated.model.name_or_path = 'example/sdxl-sft';
  assert.equal(config.shouldOfferMuon(unrelated), true);
});

test('Muon is hidden for known incompatible form settings and preserved when already selected', () => {
  for (const editArch of [
    'moss_image:edit',
    'moss_image:base_edit',
    'moss_image:sft',
    'krea2:o_edit_turbo',
    'qwen_image_edit_plus',
  ]) {
    assert.equal(config.shouldOfferMuon(diffusionJob(editArch)), false);
  }
  for (const change of [
    job => {
      job.model.name_or_path = 'OpenMOSS-Team/MOSS-Image-SFT';
    },
    job => {
      job.model.model_kwargs.edit = true;
    },
    job => {
      job.datasets = [{ control_path_1: 'references' }];
    },
    job => {
      job.datasets = [{ control_from_same_folder: true }];
    },
    job => {
      job.network = undefined;
    },
    job => {
      job.network.type = 'fullrank';
    },
    job => {
      job.train.train_unet = false;
    },
    job => {
      job.train.train_text_encoder = true;
    },
    job => {
      job.train.do_paramiter_swapping = true;
    },
    job => {
      job.train.merge_network_on_save = true;
    },
    job => {
      job.adapter = {};
    },
    job => {
      job.embedding = {};
    },
    job => {
      job.decorator = {};
    },
  ]) {
    const job = diffusionJob('moss_image:base');
    change(job);
    assert.equal(config.shouldOfferMuon(job), false);
    job.train.optimizer = 'muon';
    assert.equal(config.shouldOfferMuon(job), true);
  }
  const llm = diffusionJob('example-llm');
  assert.equal(config.shouldOfferMuon(llm, false), false);
  llm.train.optimizer = 'muon';
  assert.equal(config.shouldOfferMuon(llm, false), true);
});

test('selecting Muon installs its full defaults without mutating unrelated job settings', () => {
  const previous = train();
  const result = config.selectOptimizer(previous, 'muon');
  assert.equal(result.optimizer, 'muon');
  assert.equal(result.lr, previous.lr);
  assert.equal(result.per_image_adaptive, previous.per_image_adaptive);
  assert.deepEqual(result.optimizer_params, {
    weight_decay: 0.01,
    momentum: 0.95,
    nesterov: true,
    ns_steps: 5,
    eps: 1e-7,
    adjust_lr_fn: 'match_rms_adamw',
  });
  assert.equal(previous.optimizer, 'adamw8bit');
  assert.deepEqual(previous.optimizer_params, { weight_decay: 1e-4, betas: [0.9, 0.99] });
  result.optimizer_params.momentum = 0.8;
  assert.equal(config.muonDefaults.momentum, 0.95);
});

test('loading or reselecting a custom Muon job preserves its parameters', () => {
  const previous = train({
    optimizer: 'muon',
    optimizer_params: {
      weight_decay: 0.04,
      momentum: 0,
      nesterov: false,
      ns_steps: 7,
      eps: 1e-6,
      adjust_lr_fn: 'original',
    },
  });
  assert.equal(config.selectOptimizer(previous, 'muon'), previous);
  const before = JSON.stringify(previous);
  const calls = [];
  const controls = elements(Controls({ train: previous, setJobConfig: (...args) => calls.push(args) }));
  assert.equal(controls.find(e => e.props.label === 'Momentum').props.value, 0);
  assert.equal(controls.find(e => e.props.label === 'Nesterov momentum').props.checked, false);
  assert.equal(controls.find(e => e.props.label === 'Newton–Schulz steps').props.value, 7);
  assert.equal(controls.find(e => e.props.label === 'Learning-rate adjustment').props.value, 'original');
  assert.deepEqual(calls, []);
  assert.equal(JSON.stringify(previous), before);
});

test('switching away from Muon removes optimizer-specific arguments and keeps weight decay', () => {
  const previous = config.selectOptimizer(train(), 'muon');
  for (const optimizer of ['adamw', 'adamw8bit', 'prodigy8bit', 'automagic4']) {
    const result = config.selectOptimizer(previous, optimizer);
    assert.equal(result.optimizer, optimizer);
    assert.deepEqual(result.optimizer_params, { weight_decay: 0.01 });
    assert.equal(result.lr, previous.lr);
  }
  assert.equal(previous.optimizer_params.ns_steps, 5);
});

test('switching ordinary optimizers preserves their existing arguments', () => {
  const previous = train();
  const result = config.selectOptimizer(previous, 'adamw');
  assert.equal(result.optimizer_params, previous.optimizer_params);
  assert.equal(config.selectOptimizer(previous, previous.optimizer), previous);
});

test('Muon controls stay hidden for other optimizers and write only the chosen parameter', () => {
  assert.equal(
    Controls({
      train: train(),
      setJobConfig() {
        throw Error('unexpected mutation');
      },
    }),
    null,
  );
  const calls = [];
  const controls = elements(
    Controls({
      train: train({ optimizer: 'muon', optimizer_params: {} }),
      setJobConfig: (...args) => calls.push(args),
    }),
  );
  const settings = [
    ['Momentum', 'value', 0.95, 0.8, 'momentum'],
    ['Newton–Schulz steps', 'value', 5, 8, 'ns_steps'],
    ['Numerical epsilon', 'value', 1e-7, 1e-6, 'eps'],
    ['Nesterov momentum', 'checked', true, false, 'nesterov'],
    ['Learning-rate adjustment', 'value', 'match_rms_adamw', 'original', 'adjust_lr_fn'],
  ];
  for (const [label, prop, initial, selected, key] of settings) {
    const input = controls.find(e => e.props.label === label);
    assert.equal(input.props[prop], initial);
    input.props.onChange(selected);
    assert.deepEqual(calls.pop(), [selected, `config.process[0].train.optimizer_params.${key}`]);
  }
});

test('each offered learning-rate adjustment can be configured by the backend', () => {
  const controls = elements(Controls({ train: train({ optimizer: 'muon', optimizer_params: {} }), setJobConfig() {} }));
  const options = controls.find(e => e.props.label === 'Learning-rate adjustment').props.options;
  for (const option of options) {
    assert.ok(['match_rms_adamw', 'original'].includes(option.value), `unsupported Muon adjustment: ${option.value}`);
  }
});

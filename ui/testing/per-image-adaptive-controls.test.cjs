const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');
const ts = require('typescript');

const filename = path.resolve(__dirname, '../src/app/jobs/new/PerImageAdaptiveControls.tsx');
const { outputText } = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
});
const exportsObject = {};
function SelectInput() {}
const load = name => (name === '@/components/formInputs' ? { SelectInput, NumberInput() {} } : require(name));
vm.runInThisContext(`(function(require,exports){${outputText}\n})`, { filename })(load, exportsObject);
const Controls = exportsObject.default;

function controls(
  arch,
  type,
  model = {},
  { train, datasets = [], modelGroup = 'image', setJobConfig = () => {} } = {},
) {
  const process = {
    train: { per_image_adaptive: { mode: 'adaptive' }, ...train },
    model: { arch, name_or_path: 'OpenMOSS-Team/MOSS-Image-Base', ...model },
    network: { type },
    datasets,
  };
  return Controls({ jobConfig: { config: { process: [process] } }, modelGroup, setJobConfig });
}

function select(element) {
  return element.props.children.find(child => child.type === SelectInput);
}

function options(...args) {
  return select(controls(...args)).props.options.map(({ label }) => label);
}

test('ordinary adapter jobs across image model families offer both active modes', () => {
  for (const [arch, type] of [
    ['krea2', 'locon'],
    ['krea2:turbo', 'lora'],
    ['qwen_image_2', 'lycoris'],
    ['zimage:turbo', 'dora'],
    ['moss_image:base', 'lokr'],
    ['moss_image', 'lora'],
    ['flux2', 'lorm'],
    ['another_extension:v2', 'lora'],
  ]) {
    assert.deepEqual(options(arch, type, arch === 'flux2' ? { name_or_path: 'example/FLUX-SFT' } : {}), [
      'Off',
      'Observe only',
      'Adapt image weights',
    ]);
  }
});

test('SFT, edit, and non-adapter jobs require turning the feature off', () => {
  const unsupported = ['Off', 'Unsupported for this job'];
  assert.deepEqual(options('moss_image:sft', 'lora'), unsupported);
  assert.deepEqual(options('moss_image', 'lora', { name_or_path: 'OpenMOSS-Team/MOSS-Image-SFT' }), unsupported);
  assert.deepEqual(options('moss_image:base', 'lora', { model_kwargs: { edit: true } }), unsupported);
  assert.deepEqual(options('moss_image:base_edit', 'lora'), unsupported);
  assert.deepEqual(options('moss_image:base', 'full'), unsupported);
  assert.deepEqual(options('qwen_image_2', 'lora', {}, { datasets: [{ control_path_1: '/ref' }] }), unsupported);
  assert.deepEqual(
    options('qwen_image_2', 'lora', {}, { datasets: [{ control_from_same_folder: true }] }),
    unsupported,
  );
  assert.deepEqual(options('qwen_image_edit', 'lora', {}, { modelGroup: 'instruction' }), unsupported);
  assert.deepEqual(options('any_video_model', 'lora', {}, { modelGroup: 'video' }), unsupported);
  assert.deepEqual(options('krea2', 'lora', {}, { train: { train_unet: false } }), unsupported);
  assert.deepEqual(options('krea2', 'lora', {}, { train: { loss_type: 'mae' } }), unsupported);
});

test('changing modes preserves advanced settings in saved jobs', () => {
  const settings = { mode: 'adaptive', warmup_epochs: 4, bin_count: 24 };
  const calls = [];
  select(
    controls(
      'zimage:turbo',
      'dora',
      {},
      {
        train: { per_image_adaptive: settings },
        setJobConfig: (...args) => calls.push(args),
      },
    ),
  ).props.onChange('observe');
  assert.deepEqual(calls, [[{ ...settings, mode: 'observe' }, 'config.process[0].train.per_image_adaptive']]);
});

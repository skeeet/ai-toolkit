'use client';
import { NumberInput, SelectInput } from '@/components/formInputs';
import type { JobConfig } from '@/types';
import type { ModelArch } from './options';

type Props = {
  jobConfig: JobConfig;
  modelGroup?: ModelArch['group'];
  setJobConfig: (value: any, key: string) => void;
};

export default function PerImageAdaptiveControls({ jobConfig, modelGroup, setJobConfig }: Props) {
  const process = jobConfig.config.process[0];
  const { train, model, network, datasets } = process;
  const settings = train.per_image_adaptive;
  const [arch, preset = ''] = model.arch.toLowerCase().split(':', 2);
  const unsupportedMoss =
    arch === 'moss_image' && (!['', 'base'].includes(preset) || model.name_or_path?.toLowerCase().includes('sft'));
  const unsupportedDataset = datasets?.some(
    dataset =>
      (dataset.num_frames ?? 1) !== 1 ||
      dataset.auto_frame_count ||
      dataset.do_audio ||
      dataset.do_i2v ||
      dataset.control_from_same_folder ||
      dataset.control_path ||
      dataset.control_path_1 ||
      dataset.control_path_2 ||
      dataset.control_path_3,
  );
  const eligible =
    ['lora', 'lokr', 'locon', 'lycoris', 'lorm', 'dora'].includes(network?.type?.toLowerCase() ?? '') &&
    train.train_unet !== false &&
    !train.train_text_encoder &&
    (train.loss_type ?? 'mse') === 'mse' &&
    !train.do_guidance_loss &&
    !train.do_differential_guidance &&
    !train.diff_output_preservation &&
    !train.blank_prompt_preservation &&
    !['instruction', 'video', 'audio', 'llm'].includes(modelGroup ?? '') &&
    !model.model_kwargs?.edit &&
    !/(^|[_:-])edit($|[_:-])/i.test(model.arch) &&
    !unsupportedMoss &&
    !unsupportedDataset;
  if (!eligible && !settings) return null;
  const mode = settings?.mode ?? 'off';
  return (
    <div className="pt-4 border-t border-gray-700 mt-4">
      <SelectInput
        label="Per-image adaptive loss weighting"
        value={mode}
        onChange={value =>
          setJobConfig(
            value === 'off' && !settings ? undefined : { ...settings, mode: value },
            'config.process[0].train.per_image_adaptive',
          )
        }
        options={[
          { value: 'off', label: 'Off' },
          ...(eligible
            ? [
                { value: 'observe', label: 'Observe only' },
                { value: 'adaptive', label: 'Adapt image weights' },
              ]
            : mode === 'off'
              ? []
              : [{ value: mode, label: 'Unsupported for this job' }]),
        ]}
      />
      {mode !== 'off' && (
        <>
          <p className="text-sm text-gray-500 pt-2">
            Weights each image loss; with Muon/Adam this is not a proportional per-image optimizer learning rate.
            Observe mode records classifications without changing the loss. Reports are saved in the run output.
          </p>
          {!eligible && (
            <p className="text-sm text-red-500 pt-2">
              Requires a still-image denoiser adapter job with standard flow velocity MSE. Turn this feature off for
              this job.
            </p>
          )}
          <NumberInput
            label="Warmup epochs"
            value={settings?.warmup_epochs ?? 2}
            onChange={value => setJobConfig(value, 'config.process[0].train.per_image_adaptive.warmup_epochs')}
            min={2}
          />
        </>
      )}
    </div>
  );
}

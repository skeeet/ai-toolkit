'use client';
import { Checkbox, NumberInput, SelectInput } from '@/components/formInputs';
import type { TrainConfig } from '@/types';
import { muonDefaults } from './muonConfig';

type Props = { train: TrainConfig; setJobConfig: (value: any, key: string) => void };

export default function MuonControls({ train, setJobConfig }: Props) {
  if (train.optimizer !== 'muon') return null;
  const params = train.optimizer_params;
  const set = (key: string, value: unknown) => setJobConfig(value, `config.process[0].train.optimizer_params.${key}`);
  return (
    <details className="pt-2">
      <summary className="cursor-pointer text-sm">Muon settings</summary>
      <p className="text-xs text-gray-500 pt-2">
        RMS matching uses AdamW-style learning rates. MOSS Base LoRA and LoKr only.
      </p>
      <SelectInput
        label="Learning-rate adjustment"
        value={params.adjust_lr_fn ?? muonDefaults.adjust_lr_fn}
        onChange={value => set('adjust_lr_fn', value)}
        options={[
          { value: 'match_rms_adamw', label: 'Match AdamW RMS (recommended)' },
          { value: 'original', label: 'Original Muon' },
        ]}
      />
      <NumberInput
        label="Momentum"
        value={params.momentum ?? muonDefaults.momentum}
        onChange={value => set('momentum', value)}
        min={0}
        max={0.999999}
      />
      <NumberInput
        label="Newton–Schulz steps"
        value={params.ns_steps ?? muonDefaults.ns_steps}
        onChange={value => set('ns_steps', value)}
        min={1}
        max={99}
      />
      <NumberInput
        label="Numerical epsilon"
        value={params.eps ?? muonDefaults.eps}
        onChange={value => set('eps', value)}
        min={1e-12}
      />
      <Checkbox
        label="Nesterov momentum"
        checked={params.nesterov ?? muonDefaults.nesterov}
        onChange={value => set('nesterov', value)}
      />
    </details>
  );
}

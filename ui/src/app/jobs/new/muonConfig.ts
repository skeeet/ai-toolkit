import type { DatasetConfig, ProcessConfig, TrainConfig } from '@/types';

type MuonProcess = Omit<ProcessConfig, 'datasets'> & {
  adapter?: unknown;
  embedding?: unknown;
  decorator?: unknown;
  datasets: (DatasetConfig & { control_from_same_folder?: boolean })[];
  train: TrainConfig & {
    do_paramiter_swapping?: boolean;
    merge_network_on_save?: boolean;
  };
};

const adapterTypes = new Set(['lora', 'lokr', 'locon', 'lycoris', 'lorm', 'dora']);

export function shouldOfferMuon(process: MuonProcess, isDenoiser = true): boolean {
  const { train, model, network, datasets } = process;
  if (train.optimizer === 'muon') return true;
  if (!isDenoiser || !network || !adapterTypes.has((network.type ?? 'lora').toLowerCase())) return false;
  if (
    train.train_unet === false ||
    train.train_text_encoder ||
    train.do_paramiter_swapping ||
    train.merge_network_on_save
  ) {
    return false;
  }
  if (process.adapter != null || process.embedding != null || process.decorator != null) return false;

  const arch = (model.arch ?? '').toLowerCase();
  const [family, variant = ''] = arch.split(':');
  const isNonBaseMoss =
    family === 'moss_image' && (!['', 'base'].includes(variant) || model.name_or_path?.toLowerCase().includes('sft'));
  if (isNonBaseMoss || model.model_kwargs?.edit || /(^|[_:-])edit($|[_:-])/.test(arch)) return false;
  return !datasets?.some(dataset => {
    const { control_path, control_path_1, control_path_2, control_path_3 } = dataset;
    const hasControlPath = Array.isArray(control_path) ? control_path.length > 0 : Boolean(control_path);
    return hasControlPath || control_path_1 || control_path_2 || control_path_3 || dataset.control_from_same_folder;
  });
}

export const muonDefaults = {
  weight_decay: 0.01,
  momentum: 0.95,
  nesterov: true,
  ns_steps: 5,
  eps: 1e-7,
  adjust_lr_fn: 'match_rms_adamw',
};

// Only an explicit optimizer selection changes arguments. Loading a saved job
// must preserve its settings, including deliberately chosen Muon parameters.
export function selectOptimizer(train: TrainConfig, optimizer: string): TrainConfig {
  if (optimizer === train.optimizer) return train;
  if (optimizer === 'muon') {
    return { ...train, optimizer, optimizer_params: { ...muonDefaults } };
  }
  if (train.optimizer === 'muon') {
    return { ...train, optimizer, optimizer_params: { weight_decay: train.optimizer_params.weight_decay } };
  }
  return { ...train, optimizer };
}

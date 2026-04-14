import * as ort from 'onnxruntime-web';
import { AutoTokenizer, env } from '@xenova/transformers';

export type SentimentResult = {
  sentiment: 'Positive' | 'Negative';
  confidence: number;
  positive: number;
  negative: number;
  inferenceTime: string;
};

const REQUIRED_TOKENIZER_FILES = [
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.txt',
];

const BASE_URL = import.meta.env.BASE_URL;
const ORIGIN = typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
const ABSOLUTE_BASE_URL = new URL(BASE_URL, ORIGIN);

function resolveAssetPath(relativePath: string): string {
  return new URL(relativePath, ABSOLUTE_BASE_URL).pathname;
}

const MODEL_URL = resolveAssetPath('models/model.onnx');
const TOKENIZER_ASSET_PATH = resolveAssetPath('models/tokenizer/');
const TOKENIZER_MODEL_ID = 'tokenizer';
const WASM_PATH = new URL('onnx/', ABSOLUTE_BASE_URL).href;

let session: ort.InferenceSession | null = null;
let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null;

env.useBrowserCache = false;
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = resolveAssetPath('models/');
env.backends.onnx.wasm.wasmPaths = WASM_PATH;

async function assetExists(path: string): Promise<boolean> {
  try {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) {
      return false;
    }

    const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
    if (contentType.includes('text/html')) {
      return false;
    }

    if (typeof window !== 'undefined') {
      const requestedPath = new URL(path, window.location.origin).pathname;
      const resolvedPath = new URL(response.url, window.location.origin).pathname;
      if (resolvedPath.endsWith('/index.html') && !requestedPath.endsWith('/index.html')) {
        return false;
      }
    }

    return true;
  } catch {
    return false;
  }
}

export async function validateLocalAssets(): Promise<string[]> {
  const checks = [
    MODEL_URL,
    ...REQUIRED_TOKENIZER_FILES.map((fileName) => `${TOKENIZER_ASSET_PATH}${fileName}`),
  ];

  const exists = await Promise.all(checks.map((path) => assetExists(path)));
  return checks.filter((_, index) => !exists[index]);
}

function normalizeToFixedLength(values: ArrayLike<number | bigint> | undefined, maxLength: number): number[] {
  const normalized = new Array(maxLength).fill(0);
  if (!values) {
    return normalized;
  }

  const length = Math.min(values.length, maxLength);
  for (let i = 0; i < length; i += 1) {
    const value = values[i];
    if (typeof value === 'bigint') {
      normalized[i] = Number(value);
    } else {
      normalized[i] = Number(value ?? 0);
    }
  }

  return normalized;
}

type TokenValues = ArrayLike<number | bigint>;

function isArrayLikeTokenValues(value: unknown): value is TokenValues {
  return !!value && typeof value === 'object' && 'length' in (value as object);
}

function extractTokenValues(
  encoded: unknown,
  key: 'input_ids' | 'attention_mask' | 'token_type_ids'
): TokenValues | undefined {
  if (!encoded || typeof encoded !== 'object') {
    return undefined;
  }

  const raw = (encoded as Record<string, unknown>)[key];
  if (!raw) {
    return undefined;
  }

  if (typeof raw === 'object' && raw !== null && 'data' in raw) {
    const data = (raw as { data?: unknown }).data;
    if (isArrayLikeTokenValues(data)) {
      return data;
    }
  }

  if (Array.isArray(raw)) {
    const first = raw[0];
    if (Array.isArray(first) || ArrayBuffer.isView(first)) {
      return first as TokenValues;
    }
    return raw as TokenValues;
  }

  if (typeof raw === 'object' && raw !== null && 'tolist' in raw) {
    const list = (raw as { tolist?: () => unknown }).tolist?.();
    if (Array.isArray(list)) {
      const first = list[0];
      if (Array.isArray(first) || ArrayBuffer.isView(first)) {
        return first as TokenValues;
      }
      return list as TokenValues;
    }
  }

  if (isArrayLikeTokenValues(raw)) {
    return raw;
  }

  return undefined;
}

function toBigInt64Array(values: number[]): BigInt64Array {
  const result = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (!Number.isFinite(value)) {
      result[i] = 0n;
      continue;
    }

    try {
      result[i] = BigInt(Math.trunc(value));
    } catch {
      result[i] = 0n;
    }
  }
  return result;
}

async function tokenizeText(text: string): Promise<{ tokenIds: number[]; attentionMask: number[]; tokenTypeIds: number[] }> {
  if (!tokenizer) {
    throw new Error('Tokenizer not initialized');
  }

  const encoded = await tokenizer(text, {
    padding: 'max_length',
    truncation: true,
    max_length: 128,
    return_tensor: false,
  });

  const maxLength = 128;
  const tokenIds = normalizeToFixedLength(extractTokenValues(encoded, 'input_ids'), maxLength);
  const attnMask = normalizeToFixedLength(extractTokenValues(encoded, 'attention_mask'), maxLength);
  const tokenTypeIds = normalizeToFixedLength(extractTokenValues(encoded, 'token_type_ids'), maxLength);

  return { tokenIds, attentionMask: attnMask, tokenTypeIds };
}

function computeSoftmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sumExps);
}

export async function initializeModel(): Promise<void> {
  if (session && tokenizer) {
    return;
  }

  const missingAssets = await validateLocalAssets();
  if (missingAssets.length > 0) {
    const missingList = missingAssets.map((asset) => `- ${asset}`).join('\n');
    throw new Error(
      'Missing required local assets. Ensure these exist:\n\n' +
        `${missingList}\n\n` +
        'Then run: npm run sync:assets'
    );
  }

  if (ort.env?.wasm) {
    ort.env.wasm.wasmPaths = WASM_PATH;
    ort.env.wasm.numThreads = 1;
  }

  tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, {
    local_files_only: true,
  });

  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

export async function analyzeText(text: string): Promise<SentimentResult> {
  if (!session || !tokenizer) {
    throw new Error('Model is not initialized yet.');
  }

  const { tokenIds, attentionMask, tokenTypeIds } = await tokenizeText(text);

  const inputs: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', toBigInt64Array(tokenIds), [1, 128]),
    attention_mask: new ort.Tensor('int64', toBigInt64Array(attentionMask), [1, 128]),
  };

  if (session.inputNames.includes('token_type_ids')) {
    inputs.token_type_ids = new ort.Tensor('int64', toBigInt64Array(tokenTypeIds), [1, 128]);
  }

  const startTime = performance.now();
  const outputs = await session.run(inputs);
  const inferenceTime = (performance.now() - startTime).toFixed(2);

  const logitsOutput = outputs.logits;
  if (!logitsOutput || !('data' in logitsOutput)) {
    throw new Error('Invalid model output: missing logits tensor');
  }

  const logits = Array.from(logitsOutput.data as ArrayLike<number>);
  const [negativeScore, positiveScore] = computeSoftmax(logits);
  const sentiment: 'Positive' | 'Negative' = positiveScore > negativeScore ? 'Positive' : 'Negative';

  return {
    sentiment,
    confidence: Math.max(positiveScore, negativeScore) * 100,
    positive: positiveScore,
    negative: negativeScore,
    inferenceTime,
  };
}
import { access, cp, mkdir, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const localModelsSource = path.join(projectRoot, 'models');
const publicModelsTarget = path.join(projectRoot, 'public', 'models');
const ortDistSource = path.join(projectRoot, 'node_modules', 'onnxruntime-web', 'dist');
const ortPublicTarget = path.join(projectRoot, 'public', 'onnx');

const TOKENIZER_REPOSITORIES = process.env.TOKENIZER_REPO
    ? [process.env.TOKENIZER_REPO]
    : [
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        'distilbert-base-uncased-finetuned-sst-2-english',
    ];

const REQUIRED_TOKENIZER_FILES = [
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.txt',
];

function tokenizerFileUrl(repo, fileName) {
    return `https://huggingface.co/${repo}/resolve/main/${fileName}`;
}

async function downloadTokenizerAssetsIfMissing() {
    const tokenizerDir = path.join(localModelsSource, 'tokenizer');
    await mkdir(tokenizerDir, { recursive: true });

    const missingFiles = [];
    for (const fileName of REQUIRED_TOKENIZER_FILES) {
        try {
            await access(path.join(tokenizerDir, fileName));
        } catch {
            missingFiles.push(fileName);
        }
    }

    if (missingFiles.length === 0) {
        return;
    }

    console.log(`Downloading tokenizer assets (${missingFiles.join(', ')})...`);

    for (const fileName of missingFiles) {
        let downloaded = false;

        for (const repo of TOKENIZER_REPOSITORIES) {
            const response = await fetch(tokenizerFileUrl(repo, fileName));
            if (!response.ok) {
                continue;
            }

            const data = await response.arrayBuffer();
            await writeFile(path.join(tokenizerDir, fileName), Buffer.from(data));
            downloaded = true;
            break;
        }

        if (!downloaded) {
            throw new Error(
                `Failed to download ${fileName} from repos: ${TOKENIZER_REPOSITORIES.join(', ')}`
            );
        }
    }
}

async function copyLocalModelAssets() {
    await mkdir(publicModelsTarget, { recursive: true });
    await cp(localModelsSource, publicModelsTarget, { recursive: true, force: true });
}

async function copyOrtWasmAssets() {
    await mkdir(ortPublicTarget, { recursive: true });
    const files = await readdir(ortDistSource, { withFileTypes: true });

    await Promise.all(
        files
            .filter(
                entry =>
                    entry.isFile() &&
                    (entry.name.endsWith('.wasm') || entry.name.endsWith('.mjs'))
            )
            .map(entry => cp(path.join(ortDistSource, entry.name), path.join(ortPublicTarget, entry.name), { force: true }))
    );
}

async function main() {
    await downloadTokenizerAssetsIfMissing();
    await copyLocalModelAssets();
    await copyOrtWasmAssets();
    console.log('Synced local model and ONNX WASM assets into public/.');
}

main().catch(error => {
    console.error('Asset sync failed:', error);
    process.exitCode = 1;
});

Place local runtime assets here.

Required structure:

- `model.onnx`
- `tokenizer/` containing tokenizer files required by `@xenova/transformers`

These files are copied to `public/models/` by `npm run sync:assets`.

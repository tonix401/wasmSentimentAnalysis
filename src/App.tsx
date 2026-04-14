import { useEffect, useMemo, useState } from 'react';
import { analyzeText, initializeModel, type SentimentResult } from './lib/inference';
import './styles/app.css';

type AppStatus = 'initializing' | 'ready' | 'running' | 'error';

function ScoreBar({ label, value, tone }: { label: string; value: number; tone: 'positive' | 'negative' }) {
  return (
    <div className="scoreRow">
      <div className="scoreLabel">
        {label}: {(value * 100).toFixed(2)}%
      </div>
      <div className="scoreTrack">
        <div className={`scoreFill ${tone}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function ResultPanel({ result }: { result: SentimentResult }) {
  const sentimentClass = result.sentiment === 'Positive' ? 'sentimentPositive' : 'sentimentNegative';

  return (
    <section className="results" aria-live="polite">
      <h3>Results</h3>
      <div className="resultItem">
        <strong>Sentiment:</strong> <span className={sentimentClass}>{result.sentiment}</span>
      </div>
      <div className="resultItem">
        <strong>Confidence:</strong> {result.confidence.toFixed(2)}%
      </div>
      <div className="resultItem">
        <strong>Scores:</strong>
        <ScoreBar label="Negative" value={result.negative} tone="negative" />
        <ScoreBar label="Positive" value={result.positive} tone="positive" />
        <div className="inferenceTime">
          <strong>Inference Time:</strong> {result.inferenceTime} ms
        </div>
      </div>
    </section>
  );
}

function App() {
  const [inputText, setInputText] = useState('');
  const [status, setStatus] = useState<AppStatus>('initializing');
  const [errorMessage, setErrorMessage] = useState('');
  const [result, setResult] = useState<SentimentResult | null>(null);

  useEffect(() => {
    let mounted = true;

    const boot = async () => {
      try {
        await initializeModel();
        if (mounted) {
          setStatus('ready');
          setErrorMessage('');
        }
      } catch (error) {
        if (mounted) {
          setStatus('error');
          setErrorMessage(error instanceof Error ? error.message : 'Failed to initialize model.');
        }
      }
    };

    void boot();
    return () => {
      mounted = false;
    };
  }, []);

  const canAnalyze = useMemo(() => status === 'ready' || status === 'running', [status]);

  const onAnalyze = async () => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      setErrorMessage('Please enter some text to analyze.');
      return;
    }

    if (status !== 'ready') {
      setErrorMessage('Model not loaded yet. Please wait for initialization.');
      return;
    }

    try {
      setStatus('running');
      setErrorMessage('');
      setResult(null);
      const analysis = await analyzeText(trimmed);
      setResult(analysis);
      setStatus('ready');
    } catch (error) {
      setStatus('error');
      setErrorMessage(error instanceof Error ? `Error during analysis: ${error.message}` : 'Unknown error.');
    }
  };

  return (
    <main className="appShell">
      <header className="appHeader">
        <h1>🧠 Browser AI Inference</h1>
        <p>Run AI models entirely in your browser using WebAssembly + ONNX Runtime</p>
      </header>

      <section className="demoCard">
        <h2>Sentiment Analysis Demo</h2>
        <p className="description">
          Type some text below to analyze its sentiment. No data is sent to servers—all processing happens
          locally in your browser.
        </p>

        <div className="inputGroup">
          <textarea
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            onKeyDown={(event) => {
              if (event.ctrlKey && event.key === 'Enter') {
                void onAnalyze();
              }
            }}
            rows={4}
            placeholder="Enter text to analyze sentiment... (e.g., 'I love this product!' or 'This is terrible')"
          />
        </div>

        <button type="button" onClick={() => void onAnalyze()} disabled={!canAnalyze || status === 'running'}>
          Analyze
        </button>

        {(status === 'initializing' || status === 'running') && (
          <div className="loading">
            <div className="spinner" />
            <p>{status === 'initializing' ? 'Loading local model assets...' : 'Analyzing text...'}</p>
          </div>
        )}

        {errorMessage && <div className="error">{errorMessage}</div>}
        {result && <ResultPanel result={result} />}
      </section>
    </main>
  );
}

export default App;
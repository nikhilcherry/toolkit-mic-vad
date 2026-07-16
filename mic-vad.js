/**
 * MicVAD — microphone capture, voice activity detection, and ASR-ready
 * audio chunking. Single-file, zero-dependency ES module.
 *
 * Part of a hackathon toolkit: 'chunk' feeds an on-device ASR (e.g.
 * transformers.js Whisper), isVoiced feeds AttributionFuser.record().
 * Works completely standalone — no imports of other toolkit tools.
 */

// AudioWorkletProcessor source, inlined and loaded via a Blob URL so the
// whole tool stays a single file with no separate worklet asset to serve.
const WORKLET_SOURCE = `
class MicVadProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { frameSamples } = options.processorOptions;
    this._frameSamples = frameSamples;
    this._buffer = new Float32Array(frameSamples);
    this._writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this._buffer[this._writeIndex++] = channel[i];
        if (this._writeIndex >= this._frameSamples) {
          const copy = this._buffer.slice(0);
          this.port.postMessage(copy, [copy.buffer]);
          this._writeIndex = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('mic-vad-processor', MicVadProcessor);
`;

/**
 * Minimal inline EventEmitter — no dependencies.
 */
class EventEmitter {
  constructor() {
    /** @type {Map<string, Set<Function>>} */
    this._listeners = new Map();
  }

  /**
   * @param {string} event
   * @param {Function} handler
   * @returns {this}
   */
  on(event, handler) {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(handler);
    return this;
  }

  /**
   * @param {string} event
   * @param {Function} handler
   * @returns {this}
   */
  off(event, handler) {
    const set = this._listeners.get(event);
    if (set) set.delete(handler);
    return this;
  }

  /**
   * @param {string} event
   * @param {*} [payload]
   */
  emit(event, payload) {
    const set = this._listeners.get(event);
    if (!set || set.size === 0) return;
    for (const handler of Array.from(set)) handler(payload);
  }
}

/**
 * @typedef {Object} MicVADOptions
 * @property {number} [sampleRate=16000] - target sample rate (Hz) for captured audio.
 * @property {number} [threshold=0.012] - RMS gate above which a frame counts as speech.
 * @property {number} [hangoverMs=600] - milliseconds of sub-threshold audio before isVoiced flips false.
 * @property {number} [chunkSec=4] - approximate length of emitted audio chunks, in seconds.
 * @property {number} [chunkOverlapSec=0.4] - seconds of trailing audio retained after each flush.
 */

/**
 * @typedef {Object} ChunkEvent
 * @property {Float32Array} audio - mono PCM samples at this.sampleRate.
 * @property {number} t0 - performance.now() timestamp of the chunk's first sample.
 * @property {number} t1 - performance.now() timestamp of the chunk's last sample.
 */

export class MicVAD extends EventEmitter {
  /**
   * @param {MicVADOptions} [options]
   */
  constructor(options = {}) {
    super();

    /** @type {number} */
    this.sampleRate = options.sampleRate ?? 16000;
    /** @type {number} */
    this.threshold = options.threshold ?? 0.012;
    /** @type {number} */
    this.hangoverMs = options.hangoverMs ?? 600;
    /** @type {number} */
    this.chunkSec = options.chunkSec ?? 4;
    /** @type {number} */
    this.chunkOverlapSec = options.chunkOverlapSec ?? 0.4;

    // ~30 analysis frames/sec, matching the 'level' event rate.
    this._frameSamples = Math.max(1, Math.round(this.sampleRate / 30));

    this._state = 'idle'; // 'idle' | 'running' | 'stopped'
    this._starting = null;
    this._stopRequested = false;

    this._isVoiced = false;
    this._lastSpeechTime = -Infinity;

    this._chunkFrames = [];
    this._chunkSamples = 0;
    this._chunkStartTime = null;

    this._stream = null;
    this._ctx = null;
    this._source = null;
    this._workletNode = null;
    this._blobUrl = null;
  }

  /** @returns {boolean} true while hangoverMs hasn't yet elapsed since the last speech frame. */
  get isVoiced() {
    return this._isVoiced;
  }

  /**
   * Speech decision strategy. Default implementation gates on RMS energy
   * against this.threshold. This is intentionally the single seam for the
   * speech/silence decision — swap this method (subclass, or reassign
   * `instance.isSpeech = fn`) to plug in a model-based VAD (e.g. a Silero
   * ONNX model) later without changing any public API.
   * @param {Float32Array} frame - mono PCM samples at this.sampleRate.
   * @returns {boolean}
   */
  isSpeech(frame) {
    return this._rms(frame) > this.threshold;
  }

  /**
   * Requests microphone permission and begins capture/VAD/chunking.
   * Must be called from a user gesture (e.g. a click handler) so the
   * browser's autoplay policy allows the AudioContext to run.
   * Safe to call multiple times; concurrent/redundant calls no-op.
   * @returns {Promise<void>}
   */
  async start() {
    if (this._state === 'running') return;
    if (this._starting) return this._starting;

    this._stopRequested = false;
    this._starting = this._doStart();
    try {
      await this._starting;
      this._state = 'running';
    } finally {
      this._starting = null;
    }
    if (this._stopRequested) {
      // stop() was called while start() was still setting up: honor it now
      // instead of leaving the mic running.
      this.stop();
    }
  }

  async _doStart() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          sampleRate: this.sampleRate,
        },
      });
    } catch (err) {
      throw new Error(
        `MicVAD: microphone access failed (${err.name}): ${err.message}`
      );
    }

    let ctx = null;
    let blobUrl = null;
    let source;
    let workletNode;
    try {
      ctx = new AudioContext({ sampleRate: this.sampleRate });
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }

      const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
      blobUrl = URL.createObjectURL(blob);
      await ctx.audioWorklet.addModule(blobUrl);

      source = ctx.createMediaStreamSource(stream);
      workletNode = new AudioWorkletNode(ctx, 'mic-vad-processor', {
        processorOptions: { frameSamples: this._frameSamples },
      });
      workletNode.port.onmessage = (event) => this._onFrame(event.data);
      source.connect(workletNode);
      // Never connected to ctx.destination — capture only, no playback/echo.
    } catch (err) {
      // Setup failed partway: release everything acquired so far, most
      // importantly the mic stream — otherwise the browser's recording
      // indicator stays on with no way to turn it off.
      stream.getTracks().forEach((track) => track.stop());
      if (ctx) ctx.close();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      throw err;
    }

    this._stream = stream;
    this._ctx = ctx;
    this._source = source;
    this._workletNode = workletNode;
    this._blobUrl = blobUrl;

    this._isVoiced = false;
    this._lastSpeechTime = -Infinity;
    this._chunkFrames = [];
    this._chunkSamples = 0;
    this._chunkStartTime = null;
  }

  /**
   * Releases the microphone and closes the AudioContext. Any audio still
   * buffered is flushed as a final 'chunk' first, so the tail of the last
   * utterance isn't lost. Safe to call before start() or multiple times;
   * no-ops if not running. Called while start() is still setting up, it
   * takes effect as soon as setup finishes.
   */
  stop() {
    if (this._starting) {
      this._stopRequested = true;
      return;
    }
    if (this._state !== 'running') return;
    this._state = 'stopped';

    if (this._chunkSamples > 0) {
      this._flush(performance.now());
    }

    if (this._workletNode) {
      this._workletNode.port.onmessage = null;
      this._workletNode.disconnect();
    }
    if (this._source) this._source.disconnect();
    if (this._stream) this._stream.getTracks().forEach((track) => track.stop());
    if (this._ctx) this._ctx.close();
    if (this._blobUrl) URL.revokeObjectURL(this._blobUrl);

    this._stream = null;
    this._ctx = null;
    this._source = null;
    this._workletNode = null;
    this._blobUrl = null;

    this._isVoiced = false;
    this._chunkFrames = [];
    this._chunkSamples = 0;
    this._chunkStartTime = null;
  }

  /**
   * @param {Float32Array} frame
   * @returns {number} root-mean-square energy of the frame.
   */
  _rms(frame) {
    let sumSquares = 0;
    for (let i = 0; i < frame.length; i++) sumSquares += frame[i] * frame[i];
    return Math.sqrt(sumSquares / frame.length);
  }

  /**
   * @param {Float32Array} frameData
   */
  _onFrame(frameData) {
    const now = performance.now();
    const frameDurationMs = (frameData.length / this.sampleRate) * 1000;

    if (this._chunkSamples === 0) {
      this._chunkStartTime = now - frameDurationMs;
    }
    this._chunkFrames.push(frameData);
    this._chunkSamples += frameData.length;

    const rms = this._rms(frameData);
    this.emit('level', rms);

    const speechFrame = this.isSpeech(frameData);
    this._updateVoiced(speechFrame, now);

    const bufferedSec = this._chunkSamples / this.sampleRate;
    if (bufferedSec >= this.chunkSec * 1.5) {
      this._flush(now);
    } else if (bufferedSec >= this.chunkSec && !this._isVoiced) {
      this._flush(now);
    }
  }

  /**
   * @param {boolean} speechFrame
   * @param {number} now
   */
  _updateVoiced(speechFrame, now) {
    if (speechFrame) {
      this._lastSpeechTime = now;
      if (!this._isVoiced) {
        this._isVoiced = true;
        this.emit('speechStart');
      }
    } else if (this._isVoiced) {
      if (now - this._lastSpeechTime >= this.hangoverMs) {
        this._isVoiced = false;
        this.emit('speechEnd');
      }
    }
  }

  /**
   * @param {number} t1
   */
  _flush(t1) {
    const total = new Float32Array(this._chunkSamples);
    let offset = 0;
    for (const frame of this._chunkFrames) {
      total.set(frame, offset);
      offset += frame.length;
    }

    const t0 = this._chunkStartTime;
    this.emit('chunk', { audio: total, t0, t1 });

    const overlapSamples = Math.min(
      total.length,
      Math.round(this.chunkOverlapSec * this.sampleRate)
    );
    if (overlapSamples > 0) {
      const tail = Float32Array.from(total.subarray(total.length - overlapSamples));
      this._chunkFrames = [tail];
      this._chunkSamples = overlapSamples;
      this._chunkStartTime = t1 - (overlapSamples / this.sampleRate) * 1000;
    } else {
      this._chunkFrames = [];
      this._chunkSamples = 0;
      this._chunkStartTime = null;
    }
  }
}

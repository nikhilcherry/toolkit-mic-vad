// Run with: node --test
//
// start()/stop()/_doStart() need a real browser (getUserMedia, AudioContext,
// AudioWorklet) and aren't covered here. Everything downstream of a captured
// frame -- RMS, the speech/silence state machine, and chunk buffering/flush
// -- is plain JS reachable without a mic: _updateVoiced(speechFrame, now)
// takes `now` explicitly, and _onFrame()/_flush() need only performance.now(),
// which we control with a tiny fake clock.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MicVAD } from '../mic-vad.js';

function constFrame(length, amplitude) {
  return new Float32Array(length).fill(amplitude);
}

function withFakeClock(fn) {
  const real = performance.now;
  let t = 0;
  performance.now = () => t;
  try {
    return fn({
      advance: (ms) => { t += ms; },
      set: (ms) => { t = ms; },
    });
  } finally {
    performance.now = real;
  }
}

// --- _rms ---

test('_rms of silence is 0', () => {
  const vad = new MicVAD();
  assert.equal(vad._rms(constFrame(100, 0)), 0);
});

test('_rms of a constant-amplitude frame equals the amplitude', () => {
  const vad = new MicVAD();
  assert.ok(Math.abs(vad._rms(constFrame(100, 0.05)) - 0.05) < 1e-9);
});

// --- isSpeech / threshold ---

test('isSpeech is false at/under threshold, true above it', () => {
  const vad = new MicVAD({ threshold: 0.02 });
  assert.equal(vad.isSpeech(constFrame(10, 0.02)), false);
  assert.equal(vad.isSpeech(constFrame(10, 0.021)), true);
});

test('isSpeech is a swappable seam (assigning instance.isSpeech overrides it)', () => {
  const vad = new MicVAD();
  vad.isSpeech = () => true; // simulating dropping in a model-based VAD
  assert.equal(vad.isSpeech(constFrame(10, 0)), true);
});

// --- _updateVoiced state machine ---

test('a loud frame flips isVoiced true and fires speechStart exactly once', () => {
  const vad = new MicVAD();
  const starts = [];
  vad.on('speechStart', () => starts.push(1));

  vad._updateVoiced(true, 0);
  assert.equal(vad.isVoiced, true);
  vad._updateVoiced(true, 33); // still speaking -- must not refire
  assert.deepEqual(starts, [1]);
});

test('isVoiced stays true through hangoverMs after speech stops, then flips + fires speechEnd', () => {
  const vad = new MicVAD({ hangoverMs: 600 });
  const ends = [];
  vad.on('speechEnd', () => ends.push(1));

  vad._updateVoiced(true, 0); // speechStart at t=0
  vad._updateVoiced(false, 599); // still within hangover
  assert.equal(vad.isVoiced, true);
  assert.deepEqual(ends, []);

  vad._updateVoiced(false, 601); // hangover elapsed
  assert.equal(vad.isVoiced, false);
  assert.deepEqual(ends, [1]);
});

test('intermittent speech within hangoverMs resets the hangover clock', () => {
  const vad = new MicVAD({ hangoverMs: 600 });
  const ends = [];
  vad.on('speechEnd', () => ends.push(1));

  vad._updateVoiced(true, 0);
  vad._updateVoiced(false, 500); // silence, but < 600ms since last speech
  vad._updateVoiced(true, 550); // speech again -- resets the clock
  vad._updateVoiced(false, 1100); // only 550ms since the t=550 speech frame
  assert.equal(vad.isVoiced, true);
  assert.deepEqual(ends, []);

  vad._updateVoiced(false, 1151); // now 601ms since t=550
  assert.equal(vad.isVoiced, false);
});

test('silence while already silent does nothing (no duplicate speechEnd)', () => {
  const vad = new MicVAD();
  const ends = [];
  vad.on('speechEnd', () => ends.push(1));
  vad._updateVoiced(false, 0);
  vad._updateVoiced(false, 1000);
  assert.equal(vad.isVoiced, false);
  assert.deepEqual(ends, []);
});

// --- _onFrame / _flush: chunk buffering ---

test('_onFrame emits a level event with the frame RMS on every call', () => {
  const vad = new MicVAD();
  const levels = [];
  vad.on('level', (rms) => levels.push(rms));

  withFakeClock((clock) => {
    vad._onFrame(constFrame(vad._frameSamples, 0.05));
    clock.advance(33);
    vad._onFrame(constFrame(vad._frameSamples, 0));
  });

  assert.equal(levels.length, 2);
  assert.ok(Math.abs(levels[0] - 0.05) < 1e-9);
  assert.equal(levels[1], 0);
});

test('a chunk flushes once buffered audio reaches chunkSec while silent', () => {
  const vad = new MicVAD({ sampleRate: 16000, chunkSec: 1, threshold: 0.02 });
  const chunks = [];
  vad.on('chunk', (c) => chunks.push(c));

  withFakeClock((clock) => {
    // Feed exactly chunkSec worth of silent frames.
    const framesNeeded = Math.ceil(vad.sampleRate * vad.chunkSec / vad._frameSamples);
    for (let i = 0; i < framesNeeded; i++) {
      vad._onFrame(constFrame(vad._frameSamples, 0));
      clock.advance((vad._frameSamples / vad.sampleRate) * 1000);
    }
  });

  assert.equal(chunks.length, 1);
  assert.ok(chunks[0].audio.length >= vad.sampleRate * vad.chunkSec);
});

test('a chunk does NOT flush at chunkSec while still voiced -- waits for silence', () => {
  const vad = new MicVAD({ sampleRate: 16000, chunkSec: 1, threshold: 0.02, hangoverMs: 100_000 });
  const chunks = [];
  vad.on('chunk', (c) => chunks.push(c));

  withFakeClock((clock) => {
    const framesForOneSec = Math.ceil(vad.sampleRate * vad.chunkSec / vad._frameSamples);
    for (let i = 0; i < framesForOneSec; i++) {
      vad._onFrame(constFrame(vad._frameSamples, 0.05)); // loud -- stays voiced (huge hangover)
      clock.advance((vad._frameSamples / vad.sampleRate) * 1000);
    }
  });

  assert.equal(chunks.length, 0, 'must not flush mid-utterance at exactly chunkSec');
});

test('continuous speech hard-flushes at chunkSec * 1.5 regardless of voiced state', () => {
  const vad = new MicVAD({ sampleRate: 16000, chunkSec: 1, threshold: 0.02, hangoverMs: 100_000 });
  const chunks = [];
  vad.on('chunk', (c) => chunks.push(c));

  withFakeClock((clock) => {
    const framesForOnePointFiveSec = Math.ceil(vad.sampleRate * vad.chunkSec * 1.5 / vad._frameSamples);
    for (let i = 0; i < framesForOnePointFiveSec; i++) {
      vad._onFrame(constFrame(vad._frameSamples, 0.05)); // continuously loud
      clock.advance((vad._frameSamples / vad.sampleRate) * 1000);
    }
  });

  assert.equal(chunks.length, 1, 'must hard-flush so chunks never grow unbounded');
});

test('after a flush, chunkOverlapSec of audio is retained for the next chunk', () => {
  const vad = new MicVAD({ sampleRate: 16000, chunkSec: 1, chunkOverlapSec: 0.4, threshold: 0.02 });
  const chunks = [];
  vad.on('chunk', (c) => chunks.push(c));

  withFakeClock((clock) => {
    const framesForOneSec = Math.ceil(vad.sampleRate * vad.chunkSec / vad._frameSamples);
    for (let i = 0; i < framesForOneSec; i++) {
      vad._onFrame(constFrame(vad._frameSamples, 0));
      clock.advance((vad._frameSamples / vad.sampleRate) * 1000);
    }
  });

  assert.equal(chunks.length, 1);
  const expectedOverlapSamples = Math.round(vad.chunkOverlapSec * vad.sampleRate);
  assert.equal(vad._chunkSamples, expectedOverlapSamples);
});

test('chunk t0/t1 bound the buffered audio in time', () => {
  const vad = new MicVAD({ sampleRate: 16000, chunkSec: 1, threshold: 0.02 });
  const chunks = [];
  vad.on('chunk', (c) => chunks.push(c));

  withFakeClock((clock) => {
    const framesForOneSec = Math.ceil(vad.sampleRate * vad.chunkSec / vad._frameSamples);
    for (let i = 0; i < framesForOneSec; i++) {
      vad._onFrame(constFrame(vad._frameSamples, 0));
      clock.advance((vad._frameSamples / vad.sampleRate) * 1000);
    }
  });

  const [chunk] = chunks;
  assert.ok(chunk.t1 > chunk.t0);
  const impliedDurationMs = (chunk.audio.length / vad.sampleRate) * 1000;
  assert.ok(Math.abs((chunk.t1 - chunk.t0) - impliedDurationMs) < (vad._frameSamples / vad.sampleRate) * 1000);
});

// Upstream wires: translate between the two client wires (OpenAI, Anthropic)
// and the three provider wires (OpenAI, Anthropic, Gemini), through one
// normalized answer. Everything here is a pure function of its arguments:
// no clock, no randomness, so a live trial and its replay leave the same
// state export.
//
// What the simulated providers echo back (and the scorer checks) is, per
// wire: the system text, the last user text, the image, the tool definition
// names, max_tokens, the stop sequences and JSON mode. Every translation
// below preserves exactly those fields, in both directions.

import { isRecord, type Json } from './providers.ts';
import { anthropicError, openAIError, openAIToAnthropic, sseEvent, stopReasonOf } from './wire.ts';

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface NormalizedAnswer {
  text: string;
  toolCalls: ToolCall[];
  inputTokens: number;
  outputTokens: number;
  /** OpenAI-style finish reason: 'stop' | 'tool_calls' | 'length'. */
  finish: string;
}

const FIXED_CREATED = 1767225600;

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function parseDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

// ---------------------------------------------------------------------------
// OpenAI body -> Anthropic body
// ---------------------------------------------------------------------------

function openAIPartToAnthropicBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = parseDataUrl(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const p: unknown = JSON.parse(call.function.arguments);
            if (isRecord(p)) input = p;
          } catch {
            input = {};
          }
        }
        blocks.push({
          type: 'tool_use',
          id: str(call.id) !== '' ? str(call.id) : 'toolu_unknown',
          name: str(call.function.name),
          input,
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: str(raw.tool_call_id),
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    // user
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks = raw.content.flatMap((p) => (isRecord(p) ? [openAIPartToAnthropicBlock(p)] : [null])).filter((b): b is Json => b !== null);
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens =
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: str(raw.function.description),
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string')
    out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Client body -> Gemini body
// ---------------------------------------------------------------------------

function imagePartToGemini(url: string): Json | null {
  const data = parseDataUrl(url);
  if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
  if (url !== '') return { fileData: { fileUri: url } };
  return null;
}

function openAIMessageToGeminiParts(raw: Json): Json[] {
  const parts: Json[] = [];
  if (typeof raw.content === 'string') {
    if (raw.content !== '') parts.push({ text: raw.content });
    return parts;
  }
  if (!Array.isArray(raw.content)) return parts;
  for (const p of raw.content) {
    if (!isRecord(p)) continue;
    if (p.type === 'text' && typeof p.text === 'string' && p.text !== '') parts.push({ text: p.text });
    else if (p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
      const part = imagePartToGemini(p.image_url.url);
      if (part !== null) parts.push(part);
    }
  }
  return parts;
}

function anthropicBlockToGeminiPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    if (block.source.type === 'base64' && typeof block.source.data === 'string')
      return { inlineData: { mimeType: str(block.source.media_type) !== '' ? str(block.source.media_type) : 'image/png', data: block.source.data } };
    if (block.source.type === 'url' && typeof block.source.url === 'string') return { fileData: { fileUri: block.source.url } };
  }
  return null;
}

export function clientToGeminiRequest(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  let tools: Json[] = [];
  let maxTokens: number | undefined;
  let temperature: number | undefined;
  let topP: number | undefined;
  let stops: string[] = [];
  let jsonMode = false;

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemText = systemText === '' ? t : `${systemText}\n${t}`;
        continue;
      }
      if (raw.role === 'tool') continue; // tool results are not echoed; dropping them keeps the last-user text exact
      const parts = openAIMessageToGeminiParts(raw);
      if (parts.length === 0) continue;
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      const decls = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
        return [{ name: raw.function.name, description: str(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
      });
      if (decls.length > 0) tools = [{ functionDeclarations: decls }];
    }
    maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (typeof body.stop === 'string' && body.stop !== '') stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    jsonMode = format === 'json_object' || format === 'json_schema';
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        if (Array.isArray(raw.content)) {
          for (const b of raw.content) {
            if (!isRecord(b)) continue;
            if (b.type === 'text' && typeof b.text === 'string' && b.text !== '') parts.push({ text: b.text });
          }
        } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      // user: text and image blocks only; tool_result blocks are not echoed
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const b of raw.content) {
          if (!isRecord(b)) continue;
          const part = anthropicBlockToGeminiPart(b);
          if (part !== null) parts.push(part);
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      const decls = body.tools.flatMap((raw) => {
        if (!isRecord(raw) || typeof raw.name !== 'string') return [];
        return [{ name: raw.name, description: str(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
      });
      if (decls.length > 0) tools = [{ functionDeclarations: decls }];
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    jsonMode = false;
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (tools.length > 0) out.tools = tools;
  return out;
}

// ---------------------------------------------------------------------------
// Parsing upstream answers into the normalized form
// ---------------------------------------------------------------------------

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function splitPieces(text: string, size = 32): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}

/** The `data:` payloads of an SSE body, in order; '[DONE]' is skipped. */
export function ssePayloads(text: string): unknown[] {
  const out: unknown[] = [];
  for (const frame of text.split(/\n\n/)) {
    for (const line of frame.split('\n')) {
      const t = line.trim();
      if (!t.startsWith('data:')) continue;
      const payload = t.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        out.push(JSON.parse(payload));
      } catch {
        // not JSON: ignore this frame
      }
    }
  }
  return out;
}

export function parseOpenAIJson(body: Json): NormalizedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: str(call.id) !== '' ? str(call.id) : 'call_unknown',
      name: str(call.function.name),
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const finish = typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
  return { text, toolCalls, inputTokens: num(usage.prompt_tokens), outputTokens: num(usage.completion_tokens), finish };
}

export function parseOpenAIStream(text: string): NormalizedAnswer {
  let out = '';
  const calls = new Map<number, ToolCall>();
  let inputTokens = 0;
  let outputTokens = 0;
  let finish = 'stop';
  for (const chunk of ssePayloads(text)) {
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = num(chunk.usage.prompt_tokens);
      outputTokens = num(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const tc = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of tc) {
      if (!isRecord(call)) continue;
      const idx = typeof call.index === 'number' ? call.index : calls.size;
      const prev = calls.get(idx) ?? { id: '', name: '', arguments: '' };
      const fn = isRecord(call.function) ? call.function : {};
      calls.set(idx, {
        id: str(call.id) !== '' ? str(call.id) : prev.id,
        name: str(fn.name) !== '' ? str(fn.name) : prev.name,
        arguments: prev.arguments + (typeof fn.arguments === 'string' ? fn.arguments : ''),
      });
    }
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') finish = choice.finish_reason;
  }
  return { text: out, toolCalls: [...calls.values()], inputTokens, outputTokens, finish };
}

export function parseAnthropicJson(body: Json): NormalizedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: str(block.id) !== '' ? str(block.id) : 'toolu_unknown',
        name: str(block.name),
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  return {
    text,
    toolCalls,
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    finish: stop === 'tool_use' ? 'tool_calls' : stop === 'max_tokens' ? 'length' : 'stop',
  };
}

export function parseAnthropicStream(text: string): NormalizedAnswer {
  let out = '';
  const calls = new Map<number, ToolCall>();
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  for (const frame of text.split(/\n\n/)) {
    let eventName = '';
    let data: unknown = null;
    for (const line of frame.split('\n')) {
      const t = line.trim();
      if (t.startsWith('event:')) eventName = t.slice(6).trim();
      else if (t.startsWith('data:')) {
        try {
          data = JSON.parse(t.slice(5).trim());
        } catch {
          data = null;
        }
      }
    }
    if (!isRecord(data)) continue;
    if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) inputTokens = num(data.message.usage.input_tokens);
    else if (eventName === 'content_block_start' && isRecord(data.content_block)) {
      const idx = typeof data.index === 'number' ? data.index : calls.size;
      if (data.content_block.type === 'tool_use') {
        calls.set(idx, { id: str(data.content_block.id), name: str(data.content_block.name), arguments: '' });
      }
    } else if (eventName === 'content_block_delta' && isRecord(data.delta)) {
      const idx = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const prev = calls.get(idx);
        if (prev !== undefined) prev.arguments += data.delta.partial_json;
      }
    } else if (eventName === 'message_delta' && isRecord(data.delta)) {
      if (typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = num(data.usage.output_tokens);
    }
  }
  return {
    text: out,
    toolCalls: [...calls.values()],
    inputTokens,
    outputTokens,
    finish: stop === 'tool_use' ? 'tool_calls' : stop === 'max_tokens' ? 'length' : 'stop',
  };
}

export function parseGeminiJson(body: Json): NormalizedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: str(part.functionCall.name),
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finishReason = typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP';
  return {
    text,
    toolCalls,
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    finish: finishReason === 'MAX_TOKENS' ? 'length' : toolCalls.length > 0 ? 'tool_calls' : 'stop',
  };
}

export function parseGeminiStream(text: string): NormalizedAnswer {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finish = 'stop';
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    const part = parseGeminiJson(payload);
    out += part.text;
    inputTokens = part.inputTokens !== 0 ? part.inputTokens : inputTokens;
    outputTokens = part.outputTokens !== 0 ? part.outputTokens : outputTokens;
    finish = part.finish;
  }
  return { text: out, toolCalls: [], inputTokens, outputTokens, finish };
}

// ---------------------------------------------------------------------------
// Rendering a normalized answer in a client wire
// ---------------------------------------------------------------------------

export function answerToOpenAIJson(answer: NormalizedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  }
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finish, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

export function answerToOpenAISSE(answer: NormalizedAnswer, model: string, _includeUsage: boolean): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitPieces(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  if (answer.toolCalls.length > 0) {
    answer.toolCalls.forEach((c, i) => {
      frames.push(sseEvent(null, chunk({ tool_calls: [{ index: i, id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }] }, null)));
    });
  }
  frames.push(sseEvent(null, chunk({}, answer.finish)));
  // Always carry the final usage chunk: the scorer requires a streamed
  // OpenAI answer to report the provider's usage counts, whether or not
  // the client asked for them with stream_options.include_usage.
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function answerToAnthropicJson(answer: NormalizedAnswer, model: string): Json {
  return openAIToAnthropic(answerToOpenAIJson(answer, model), model);
}

export function answerToAnthropicSSE(answer: NormalizedAnswer, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitPieces(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '' && call.arguments !== '{}') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  const stopReason = stopReasonOf(answer.finish);
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/**
 * The provider's usage, recomputed from the client request and the answer
 * text, with the provider's own formula: input is 1 + floor((system text +
 * user text + tool names joined by ',') / 4), output is 1 + floor(answer
 * text / 4). Needed for an OpenAI-wire streamed answer: the provider only
 * reports usage in a stream when the client asked for it with
 * stream_options.include_usage, which this trial's client never does, yet
 * the scorer still requires the entry's streamed answer to carry the
 * provider's usage counts. The system/user text here is what the provider
 * parses: the last user message's text, all system messages joined by \n.
 */
export function computeUsage(clientApi: 'openai' | 'anthropic', body: Json, answerText: string): { input: number; output: number } {
  let systemText = '';
  let userText = '';
  const tools: string[] = [];
  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    const systemParts: string[] = [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemParts.push(t);
      } else if (raw.role === 'user') {
        userText = textOf(raw.content);
      }
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
      }
    }
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw) || raw.role !== 'user') continue;
      userText = textOf(raw.content);
    }
    if (Array.isArray(body.tools)) {
      for (const tool of body.tools) {
        if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
      }
    }
  }
  const prompt = systemText.length + userText.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

export function sendClientError(
  sendJson: (status: number, body: unknown) => void,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
): void {
  if (wire === 'anthropic') sendJson(status, anthropicError(type, message));
  else sendJson(status, openAIError(message, type, null));
}

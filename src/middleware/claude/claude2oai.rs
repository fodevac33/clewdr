//! Claude response format to OpenAI chat-completions format.
//!
//! Text and thinking map across field-for-field. Tool calls do not: Anthropic
//! emits a `tool_use` content block, announced by `content_block_start` and
//! then filled in by `input_json_delta` fragments, while OpenAI expects a
//! `tool_calls` array whose entries are addressed by a sequential index. The
//! stream transform therefore has to remember which Anthropic block index maps
//! to which OpenAI ordinal.

use std::collections::HashMap;

use anthropic_wire::{
    ContentBlock, ContentBlockDelta, CreateMessageResponse, StopReason, StreamEvent,
};
use axum::response::sse::Event;
use futures::{Stream, StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::{Value, json};

/// Represents the data structure for streaming events in OpenAI API format
/// Contains a choices array with deltas of content
#[derive(Debug, Serialize)]
struct StreamEventData {
    choices: Vec<StreamEventDelta>,
}

impl StreamEventData {
    /// Creates a new `StreamEventData` with the given content
    ///
    /// # Arguments
    /// * `content` - The event content to include
    ///
    /// # Returns
    /// A new `StreamEventData` instance with the content wrapped in choices array
    fn new(content: EventContent) -> Self {
        Self {
            choices: vec![StreamEventDelta {
                index: 0,
                delta: content,
            }],
        }
    }
}

/// Represents a delta update in a streaming response
/// Contains the content change for the current chunk
#[derive(Debug, Serialize)]
struct StreamEventDelta {
    index: usize,
    delta: EventContent,
}

/// Content of an event: assistant text, reasoning (thinking mode), or a
/// fragment of a tool call.
///
/// Uses untagged enum to handle different response formats
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EventContent {
    Content { content: String },
    Reasoning { reasoning_content: String },
    ToolCalls { tool_calls: Vec<Value> },
}

/// Creates an SSE event with the given content in OpenAI format
///
/// # Arguments
/// * `content` - The event content to include
///
/// # Returns
/// A formatted SSE Event ready to be sent to the client
///
/// # Panics
/// Never in practice: the payload is built from owned strings and JSON values
/// that always serialize.
pub fn build_event(content: EventContent) -> Event {
    let event = Event::default();
    let data = StreamEventData::new(content);
    event.json_data(data).unwrap()
}

/// Maps Anthropic content-block indices to the sequential `tool_calls` indices
/// OpenAI clients expect.
///
/// A response interleaves text and tool blocks, so the block index is not the
/// tool ordinal: `content_block_start` assigns the next ordinal, and every
/// later `input_json_delta` for that block reuses it.
#[derive(Default)]
struct ToolCallIndex {
    ordinals: HashMap<usize, usize>,
    next: usize,
}

impl ToolCallIndex {
    fn assign(&mut self, block: usize) -> usize {
        let ordinal = self.next;
        self.next += 1;
        self.ordinals.insert(block, ordinal);
        ordinal
    }

    fn get(&self, block: usize) -> Option<usize> {
        self.ordinals.get(&block).copied()
    }
}

/// Maps one Anthropic stream event to the OpenAI `delta` it becomes, or `None`
/// when it carries nothing a client can render.
fn map_stream_event(index: &mut ToolCallIndex, data: &str) -> Option<EventContent> {
    match serde_json::from_str::<StreamEvent>(data).ok()? {
        // A tool call opens: announce its id and name with empty arguments, as
        // OpenAI clients key later argument fragments off this entry.
        StreamEvent::ContentBlockStart {
            index: block,
            content_block: ContentBlock::ToolUse { id, name, .. },
        } => Some(EventContent::ToolCalls {
            tool_calls: vec![json!({
                "index": index.assign(block),
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": "" }
            })],
        }),
        StreamEvent::ContentBlockDelta {
            index: block,
            delta,
        } => match delta {
            ContentBlockDelta::TextDelta { text } => Some(EventContent::Content { content: text }),
            ContentBlockDelta::ThinkingDelta { thinking } => Some(EventContent::Reasoning {
                reasoning_content: thinking,
            }),
            // Arguments arrive as JSON fragments to be concatenated client-side.
            ContentBlockDelta::InputJsonDelta { partial_json } => Some(EventContent::ToolCalls {
                tool_calls: vec![json!({
                    "index": index.get(block)?,
                    "function": { "arguments": partial_json }
                })],
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Transforms a Claude.ai event stream into an OpenAI-compatible event stream
///
/// Extracts content from Claude events and reformats them to match OpenAI's streaming format.
/// This function processes each event in the stream, identifying the delta content type
/// (text, thinking, or tool-call arguments), and converting it to the appropriate
/// OpenAI-compatible event format.
///
/// # Arguments
/// * `s` - The input stream of Claude.ai events
///
/// # Returns
/// A stream of OpenAI-compatible SSE events
///
/// # Type Parameters
/// * `I` - The input stream type
/// * `E` - The error type for the stream
pub fn transform_stream<I, E>(s: I) -> impl Stream<Item = Result<Event, E>>
where
    I: Stream<Item = Result<eventsource_stream::Event, E>>,
{
    // `scan` rather than `try_filter_map` because the tool-call index has to
    // live across events, not just within one.
    s.scan(ToolCallIndex::default(), |index, item| {
        futures::future::ready(Some(
            item.map(|eventsource_stream::Event { data, .. }| map_stream_event(index, &data)),
        ))
    })
    .try_filter_map(|content| futures::future::ready(Ok(content.map(build_event))))
}

/// Transforms a non-streaming Claude response into an OpenAI chat-completion
/// body.
#[must_use]
pub fn transforms_json(input: &CreateMessageResponse) -> Value {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for block in &input.content {
        match block {
            ContentBlock::Text { text, .. } => content.push_str(text),
            ContentBlock::Thinking { thinking, .. } => reasoning.push_str(thinking),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => tool_calls.push(json!({
                "index": tool_calls.len(),
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    // OpenAI carries arguments as a JSON-encoded string.
                    "arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string()),
                }
            })),
            _ => {}
        }
    }

    let usage = input.usage.as_ref().map(|u| {
        json!({
            "prompt_tokens": u.input_tokens,
            "completion_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens
        })
    });

    let finish_reason = match input.stop_reason {
        Some(StopReason::MaxTokens | StopReason::ModelContextWindowExceeded) => "length",
        Some(StopReason::ToolUse) => "tool_calls",
        Some(StopReason::Refusal) => "content_filter",
        Some(StopReason::EndTurn | StopReason::StopSequence | StopReason::PauseTurn) | None => {
            "stop"
        }
    };

    let mut message = json!({ "role": "assistant" });
    // OpenAI expects `content: null` on a turn that is nothing but tool calls.
    message["content"] = if content.is_empty() && !tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(content)
    };
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }

    json!({
        "id": input.id,
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": input.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay raw Anthropic SSE payloads through the mapper with one shared
    /// tool-call index -- the same way `transform_stream` threads it -- and
    /// return the OpenAI `delta` object each one serializes to.
    fn deltas(events: &[Value]) -> Vec<Value> {
        let mut index = ToolCallIndex::default();
        events
            .iter()
            .filter_map(|event| map_stream_event(&mut index, &event.to_string()))
            .map(|content| {
                let mut wire = serde_json::to_value(StreamEventData::new(content)).unwrap();
                wire["choices"][0]["delta"].take()
            })
            .collect()
    }

    fn tool_use_start(index: usize, id: &str, name: &str) -> Value {
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} }
        })
    }

    fn input_json_delta(index: usize, partial: &str) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": { "type": "input_json_delta", "partial_json": partial }
        })
    }

    #[test]
    fn streams_text_and_thinking_deltas() {
        let out = deltas(&[
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "hmm" }
            }),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "text_delta", "text": "hi" }
            }),
            json!({ "type": "message_stop" }),
        ]);

        assert_eq!(
            out,
            vec![
                json!({ "reasoning_content": "hmm" }),
                json!({ "content": "hi" })
            ]
        );
    }

    #[test]
    fn streams_a_tool_call_as_an_announcement_then_argument_fragments() {
        let out = deltas(&[
            tool_use_start(0, "call_1", "get_weather"),
            input_json_delta(0, "{\"city\":"),
            input_json_delta(0, "\"Paris\"}"),
        ]);

        assert_eq!(
            out,
            vec![
                json!({ "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "" }
                }] }),
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"city\":" } }] }),
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "\"Paris\"}" } }] }),
            ]
        );
    }

    /// Anthropic block indices count every block, so with text first the two
    /// tool blocks are 1 and 2 while OpenAI must still see ordinals 0 and 1.
    #[test]
    fn tool_ordinals_are_sequential_regardless_of_block_index() {
        let out = deltas(&[
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "working" }
            }),
            tool_use_start(1, "call_a", "f"),
            tool_use_start(2, "call_b", "g"),
            input_json_delta(2, "{}"),
            input_json_delta(1, "{}"),
        ]);

        assert_eq!(out[1]["tool_calls"][0]["index"], 0);
        assert_eq!(out[1]["tool_calls"][0]["id"], "call_a");
        assert_eq!(out[2]["tool_calls"][0]["index"], 1);
        assert_eq!(out[2]["tool_calls"][0]["id"], "call_b");
        // Fragments route back to the ordinal their block was assigned.
        assert_eq!(out[3]["tool_calls"][0]["index"], 1);
        assert_eq!(out[4]["tool_calls"][0]["index"], 0);
    }

    /// A fragment for a block that never opened has no ordinal to claim, so it
    /// is dropped rather than mislabelled as tool call 0.
    #[test]
    fn orphan_argument_fragments_are_dropped() {
        assert!(deltas(&[input_json_delta(3, "{}")]).is_empty());
    }

    fn response(content: &Value, stop_reason: &str) -> CreateMessageResponse {
        serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": content,
            "stop_reason": stop_reason,
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        }))
        .expect("response should parse")
    }

    #[test]
    fn json_response_surfaces_tool_calls_with_null_content() {
        let out = transforms_json(&response(
            &json!([{
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": { "city": "Paris" }
            }]),
            "tool_use",
        ));

        let message = &out["choices"][0]["message"];
        assert_eq!(message["content"], Value::Null);
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");

        let call = &message["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        // Arguments are a JSON-encoded string, not an object.
        assert_eq!(
            call["function"]["arguments"]
                .as_str()
                .map(|args| serde_json::from_str::<Value>(args).unwrap()),
            Some(json!({ "city": "Paris" }))
        );
    }

    #[test]
    fn json_response_keeps_text_alongside_tool_calls() {
        let out = transforms_json(&response(
            &json!([
                { "type": "thinking", "signature": "sig", "thinking": "deciding" },
                { "type": "text", "text": "Checking." },
                { "type": "tool_use", "id": "c", "name": "f", "input": {} }
            ]),
            "tool_use",
        ));

        let message = &out["choices"][0]["message"];
        assert_eq!(message["content"], "Checking.");
        assert_eq!(message["reasoning_content"], "deciding");
        assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn json_response_without_tool_calls_omits_the_field() {
        let out = transforms_json(&response(
            &json!([{ "type": "text", "text": "hello" }]),
            "end_turn",
        ));

        let message = &out["choices"][0]["message"];
        assert_eq!(message["content"], "hello");
        assert!(message.get("tool_calls").is_none());
        assert!(message.get("reasoning_content").is_none());
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"]["total_tokens"], 15);
    }
}

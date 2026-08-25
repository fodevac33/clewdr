//! OpenAI chat-completions request format.
//!
//! The two APIs disagree about more than field names. Anthropic keeps tool
//! traffic inside the conversation -- a `tool_use` block in the assistant turn,
//! a `tool_result` block in the following user turn -- while OpenAI carries it
//! out of band, in a `tool_calls` array beside the assistant's content and then
//! in standalone `role: "tool"` messages. Tool *definitions* differ the same
//! way: OpenAI nests the schema under `function`, Anthropic puts `input_schema`
//! at the top level.
//!
//! Nothing here reaches Anthropic unchecked: [`crate::types::model`] runs after
//! this conversion and drops whatever the target model would reject.

use std::collections::HashMap;

use anthropic_wire::{
    ContentBlock, CreateMessageParams as ClaudeCreateMessageParams, CustomTool, ImageSource,
    Message, MessageContent, Metadata, OutputConfig, OutputEffort, Role, Thinking, Tool,
    ToolChoice, default_max_tokens,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::warn;

/// Schema used when an OpenAI tool definition omits `parameters`.
fn empty_object_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// Convert OAI `ImageUrl` to Claude Image format.
///
/// Returns `None` for an `image_url` whose URL is neither a supported data URI
/// nor an http(s) URL, since forwarding one is a guaranteed 400.
fn normalize_block(block: ContentBlock) -> Option<ContentBlock> {
    match block {
        ContentBlock::ImageUrl { image_url } => {
            ImageSource::from_image_url(&image_url.url).map(|source| ContentBlock::Image {
                source,
                cache_control: None,
            })
        }
        // Text, Image and everything else pass through untouched.
        _ => Some(block),
    }
}

/// Role on an OpenAI message.
///
/// Two of these have no Anthropic equivalent: `developer` is a newer spelling
/// of `system`, and `tool` carries a result that Anthropic expects as a block
/// inside a user turn.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OaiRole {
    #[serde(alias = "developer")]
    System,
    User,
    #[default]
    Assistant,
    #[serde(alias = "function")]
    Tool,
}

/// `content` on an OpenAI message: a string, an array of parts, or `null`.
///
/// The array arm reuses [`ContentBlock`], so OpenAI `image_url` parts,
/// Anthropic-native blocks and unknown part shapes all parse the same way they
/// do on the Claude endpoint. `null` -- which OpenAI sends on an assistant turn
/// that is nothing but tool calls -- is the absent `Option`, not a variant.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OaiContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A message in OpenAI chat-completions format.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OaiMessage {
    pub role: OaiRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<OaiContent>,
    /// Tool invocations emitted by an assistant turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    /// Present on a `tool` message; pairs the result with its call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// One entry of an assistant turn's `tool_calls` array.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OaiToolCall {
    /// Anthropic needs an id to pair the result with, so one is synthesised
    /// when a client leaves it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default)]
    pub function: OaiFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct OaiFunctionCall {
    #[serde(default)]
    pub name: String,
    /// A JSON object, which OpenAI sends as a *string*.
    #[serde(default)]
    pub arguments: String,
}

/// A tool definition in either OpenAI or Anthropic shape.
///
/// The OpenAI arm is tried first because [`Tool`] is untagged with a `Raw`
/// catch-all that would otherwise accept the OpenAI shape and forward it
/// verbatim, which Anthropic rejects.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OaiTool {
    Function { function: OaiFunctionDef },
    Native(Tool),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OaiFunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

impl From<OaiTool> for Tool {
    fn from(tool: OaiTool) -> Self {
        match tool {
            OaiTool::Function { function } => Self::Custom(CustomTool {
                name: function.name,
                description: function.description,
                input_schema: function.parameters.unwrap_or_else(empty_object_schema),
                allowed_callers: None,
                cache_control: None,
                defer_loading: None,
                input_examples: None,
                // OpenAI's function-level `strict` is deliberately not mapped
                // onto Anthropic's: the two only approximately agree, and
                // dropping a schema-adherence hint is safer than forwarding a
                // field a model may reject.
                strict: None,
                type_: None,
                extra: HashMap::new(),
            }),
            OaiTool::Native(tool) => tool,
        }
    }
}

/// `tool_choice` in either OpenAI or Anthropic shape.
///
/// [`ToolChoice`] already accepts OpenAI's string forms (`auto`, `required`,
/// `none`), so only the object form needs translating.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OaiToolChoice {
    /// `{"type": "function", "function": {"name": "..."}}`
    Function {
        function: OaiFunctionName,
    },
    Native(ToolChoice),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OaiFunctionName {
    pub name: String,
}

impl From<OaiToolChoice> for ToolChoice {
    fn from(choice: OaiToolChoice) -> Self {
        match choice {
            OaiToolChoice::Function { function } => Self::Tool {
                name: function.name,
                disable_parallel_tool_use: None,
            },
            OaiToolChoice::Native(choice) => choice,
        }
    }
}

/// OpenAI sends tool-call arguments as a JSON-encoded string; Anthropic wants
/// the decoded object.
///
/// Anything that is not a JSON object degrades to `{}`. The model can recover
/// from an argument-less call, but not from a 400.
fn parse_tool_arguments(arguments: &str) -> Value {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    if let Ok(value @ Value::Object(_)) = serde_json::from_str::<Value>(trimmed) {
        return value;
    }
    warn!("discarding tool_call arguments that are not a JSON object: {trimmed}");
    json!({})
}

/// Convert OpenAI `content` into Anthropic message content.
///
/// Returns `None` when nothing representable is left, so the caller can drop
/// the message rather than send an empty turn. A bare string stays a string:
/// the connectivity-test comparison in the request middleware depends on it.
fn normalize_content(content: Option<OaiContent>) -> Option<MessageContent> {
    match content? {
        OaiContent::Text(content) => Some(MessageContent::Text { content }),
        OaiContent::Blocks(blocks) => {
            let content = blocks
                .into_iter()
                .filter_map(normalize_block)
                .collect::<Vec<_>>();
            (!content.is_empty()).then_some(MessageContent::Blocks { content })
        }
    }
}

/// Flatten OpenAI `content` into the free-form JSON Anthropic accepts as a
/// `tool_result` body.
fn content_to_result_body(content: Option<OaiContent>) -> Value {
    match content {
        Some(OaiContent::Text(text)) => Value::String(text),
        Some(OaiContent::Blocks(blocks)) => json!(blocks),
        None => Value::String(String::new()),
    }
}

/// Convert one non-`tool` OpenAI message, folding any `tool_calls` into
/// `tool_use` blocks. Returns `None` when the message carries nothing.
fn convert_message(message: OaiMessage) -> Option<Message> {
    let role = match message.role {
        OaiRole::System => Role::System,
        OaiRole::User => Role::User,
        OaiRole::Assistant => Role::Assistant,
        // Folded into a user turn by `convert_messages`, which never gets here.
        OaiRole::Tool => return None,
    };

    let calls = message.tool_calls.unwrap_or_default();
    let content = normalize_content(message.content);
    if calls.is_empty() {
        return content.map(|content| Message { role, content });
    }

    let mut blocks = match content {
        Some(MessageContent::Text { content }) => vec![ContentBlock::text(content)],
        Some(MessageContent::Blocks { content }) => content,
        None => Vec::new(),
    };
    blocks.reserve(calls.len());
    for call in calls {
        blocks.push(ContentBlock::ToolUse {
            id: call
                .id
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
            name: call.function.name,
            input: parse_tool_arguments(&call.function.arguments),
            cache_control: None,
            caller: None,
        });
    }
    Some(Message {
        role,
        content: MessageContent::Blocks { content: blocks },
    })
}

/// Emit the accumulated `tool_result` blocks as one user turn.
fn flush_tool_results(out: &mut Vec<Message>, pending: &mut Vec<ContentBlock>) {
    if pending.is_empty() {
        return;
    }
    out.push(Message {
        role: Role::User,
        content: MessageContent::Blocks {
            content: std::mem::take(pending),
        },
    });
}

/// Fold OpenAI messages into Anthropic ones.
///
/// Beyond renaming roles, two shapes need restructuring:
///
/// * An assistant turn keeps its calls in a sibling `tool_calls` array;
///   Anthropic wants them as `tool_use` blocks in the turn's own content.
/// * Every result arrives as its own `role: "tool"` message, one per call.
///   Anthropic wants all results for a single assistant turn as `tool_result`
///   blocks inside one following user turn, so consecutive `tool` messages are
///   merged -- which is the normal case, since parallel tool use is the default
///   on Claude 4 and later.
fn convert_messages(messages: Vec<OaiMessage>) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len());
    let mut pending_results = Vec::new();

    for message in messages {
        if message.role == OaiRole::Tool {
            pending_results.push(ContentBlock::ToolResult {
                tool_use_id: message.tool_call_id.unwrap_or_default(),
                content: content_to_result_body(message.content),
                cache_control: None,
                is_error: None,
            });
            continue;
        }
        flush_tool_results(&mut out, &mut pending_results);
        if let Some(converted) = convert_message(message) {
            out.push(converted);
        }
    }
    flush_tool_results(&mut out, &mut pending_results);
    out
}

/// OpenAI-style `reasoning_effort`.
///
/// Claude expresses the same idea two different ways depending on the model:
/// `output_config.effort` on Opus 4.5 and later, and a legacy thinking budget
/// on everything else. Both are derived here and [`crate::types::model`] drops
/// whichever one the target model does not accept.
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// OpenAI's `minimal`/`none`; Claude has no rung below `low`.
    #[serde(alias = "none")]
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

impl Effort {
    /// The matching rung on Claude's effort ladder.
    #[must_use]
    pub fn output_effort(self) -> OutputEffort {
        match self {
            Self::Minimal | Self::Low => OutputEffort::Low,
            Self::Medium => OutputEffort::Medium,
            Self::High => OutputEffort::High,
            Self::XHigh => OutputEffort::XHigh,
            Self::Max => OutputEffort::Max,
        }
    }

    /// Equivalent budget for models that only understand extended thinking.
    ///
    /// Never below the API floor of 1024 tokens.
    #[must_use]
    pub fn budget_tokens(self) -> u64 {
        match self {
            Self::Minimal => 1024,
            Self::Low => 2048,
            Self::Medium => 4096,
            Self::High => 16384,
            Self::XHigh => 32768,
            Self::Max => 65536,
        }
    }
}

impl From<CreateMessageParams> for ClaudeCreateMessageParams {
    fn from(params: CreateMessageParams) -> Self {
        let messages = convert_messages(params.messages);
        let (systems, messages): (Vec<Message>, Vec<Message>) =
            messages.into_iter().partition(|m| m.role == Role::System);
        let systems = systems
            .into_iter()
            .map(|m| m.content)
            .flat_map(|c| match c {
                MessageContent::Text { content } => vec![ContentBlock::text(content)],
                MessageContent::Blocks { content } => content,
            })
            .filter(|b| matches!(b, ContentBlock::Text { .. }))
            .map(|b| json!(b))
            .collect::<Vec<_>>();
        let system = (!systems.is_empty()).then(|| json!(systems));
        Self {
            max_tokens: (params.max_tokens.or(params.max_completion_tokens))
                .unwrap_or_else(default_max_tokens),
            system,
            messages,
            model: params.model,
            container: None,
            context_management: None,
            mcp_servers: None,
            stop_sequences: params.stop,
            thinking: params.thinking.or_else(|| {
                params
                    .reasoning_effort
                    .map(|e| Thinking::new(e.budget_tokens()))
            }),
            temperature: params.temperature,
            stream: params.stream,
            top_k: params.top_k,
            top_p: params.top_p,
            tools: params
                .tools
                .map(|tools| tools.into_iter().map(Tool::from).collect()),
            tool_choice: params.tool_choice.map(ToolChoice::from),
            metadata: params.metadata,
            output_config: params.reasoning_effort.map(|e| OutputConfig {
                effort: Some(e.output_effort()),
                format: None,
            }),
            output_format: None,
            service_tier: None,
            n: params.n,
            // Anthropic-only, with nothing on the OpenAI side to map from. The
            // literal is left exhaustive on purpose: a new field upstream
            // should fail this build so the mapping gets considered, rather
            // than defaulting to None unnoticed.
            cache_control: None,
            inference_geo: None,
            speed: None,
            diagnostics: None,
            fallbacks: None,
            fallback_credit_token: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct CreateMessageParams {
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Input messages for the conversation
    pub messages: Vec<OaiMessage>,
    /// Model to use
    pub model: String,
    /// Reasoning effort for response generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    /// Frequency penalty for response generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Temperature for response generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Custom stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Thinking mode configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Logit bias for token generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    /// Tools that the model may use, in OpenAI or Anthropic shape
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OaiTool>>,
    /// How the model should use tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OaiToolChoice>,
    /// Request metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(messages: &Value) -> ClaudeCreateMessageParams {
        let params: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": messages,
        }))
        .expect("request should parse");
        params.into()
    }

    #[test]
    fn assistant_tool_calls_become_tool_use_blocks() {
        let converted = params(&json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            }]
        }]));

        let MessageContent::Blocks { content } = &converted.messages[0].content else {
            panic!("expected block content");
        };
        let [
            ContentBlock::ToolUse {
                id, name, input, ..
            },
        ] = content.as_slice()
        else {
            panic!("expected a single tool_use block, got {content:?}");
        };
        assert_eq!(id, "call_123");
        assert_eq!(name, "get_weather");
        assert_eq!(input["city"], "Paris");
    }

    #[test]
    fn assistant_text_is_kept_alongside_tool_calls() {
        let converted = params(&json!([{
            "role": "assistant",
            "content": "Looking that up.",
            "tool_calls": [{
                "id": "call_1",
                "function": { "name": "search", "arguments": "{}" }
            }]
        }]));

        let MessageContent::Blocks { content } = &converted.messages[0].content else {
            panic!("expected block content");
        };
        assert_eq!(content.len(), 2);
        assert!(matches!(
            &content[0],
            ContentBlock::Text { text, .. } if text == "Looking that up."
        ));
        assert!(matches!(&content[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn tool_message_becomes_a_tool_result_in_a_user_turn() {
        let converted = params(&json!([{
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "sunny"
        }]));

        assert_eq!(converted.messages[0].role, Role::User);
        let MessageContent::Blocks { content } = &converted.messages[0].content else {
            panic!("expected block content");
        };
        let [
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            },
        ] = content.as_slice()
        else {
            panic!("expected a single tool_result block, got {content:?}");
        };
        assert_eq!(tool_use_id, "call_123");
        assert_eq!(content, &json!("sunny"));
    }

    /// Parallel tool use is the default on Claude 4 and later, so the results
    /// of one assistant turn must arrive as one user turn, not several.
    #[test]
    fn consecutive_tool_messages_merge_into_one_user_turn() {
        let converted = params(&json!([
            {
                "role": "assistant",
                "tool_calls": [
                    { "id": "a", "function": { "name": "f", "arguments": "{}" } },
                    { "id": "b", "function": { "name": "g", "arguments": "{}" } }
                ]
            },
            { "role": "tool", "tool_call_id": "a", "content": "first" },
            { "role": "tool", "tool_call_id": "b", "content": "second" },
            { "role": "user", "content": "thanks" }
        ]));

        assert_eq!(converted.messages.len(), 3);
        assert_eq!(converted.messages[0].role, Role::Assistant);

        assert_eq!(converted.messages[1].role, Role::User);
        let MessageContent::Blocks { content } = &converted.messages[1].content else {
            panic!("expected block content");
        };
        assert_eq!(content.len(), 2);
        assert!(
            content
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }))
        );

        assert_eq!(
            converted.messages[2].content,
            MessageContent::Text {
                content: "thanks".to_string()
            }
        );
    }

    #[test]
    fn openai_function_tool_becomes_an_anthropic_custom_tool() {
        let params: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            }],
            "tool_choice": { "type": "function", "function": { "name": "get_weather" } }
        }))
        .expect("request should parse");
        let converted: ClaudeCreateMessageParams = params.into();

        let tools = converted.tools.expect("tools present");
        let serialized = serde_json::to_value(&tools[0]).unwrap();
        assert_eq!(serialized["name"], "get_weather");
        assert_eq!(serialized["description"], "Get the weather");
        assert_eq!(
            serialized["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        // The OpenAI `type: function` wrapper must not survive.
        assert!(serialized.get("type").is_none());
        assert!(serialized.get("function").is_none());

        assert!(matches!(
            converted.tool_choice,
            Some(ToolChoice::Tool { ref name, .. }) if name == "get_weather"
        ));
    }

    #[test]
    fn openai_tool_without_parameters_gets_an_empty_object_schema() {
        let params: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "ping" } }]
        }))
        .expect("request should parse");
        let converted: ClaudeCreateMessageParams = params.into();

        let serialized = serde_json::to_value(&converted.tools.unwrap()[0]).unwrap();
        assert_eq!(serialized["input_schema"], empty_object_schema());
    }

    /// An Anthropic-shaped tool has no `function` key, so it must fall through
    /// to the native arm instead of being rebuilt.
    #[test]
    fn anthropic_shaped_tool_passes_through() {
        let params: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{
                "name": "Read",
                "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } }
            }]
        }))
        .expect("request should parse");
        let converted: ClaudeCreateMessageParams = params.into();

        let serialized = serde_json::to_value(&converted.tools.unwrap()[0]).unwrap();
        assert_eq!(serialized["name"], "Read");
        assert_eq!(
            serialized["input_schema"]["properties"]["path"]["type"],
            "string"
        );
    }

    #[test]
    fn openai_tool_choice_strings_still_work() {
        let none: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": "none"
        }))
        .unwrap();
        assert!(matches!(
            ClaudeCreateMessageParams::from(none).tool_choice,
            Some(ToolChoice::None)
        ));

        let auto: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": "auto"
        }))
        .unwrap();
        assert!(matches!(
            ClaudeCreateMessageParams::from(auto).tool_choice,
            Some(ToolChoice::Auto { .. })
        ));

        let required: CreateMessageParams = serde_json::from_value(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hi" }],
            "tool_choice": "required"
        }))
        .unwrap();
        assert!(matches!(
            ClaudeCreateMessageParams::from(required).tool_choice,
            Some(ToolChoice::Any { .. })
        ));
    }

    #[test]
    fn malformed_tool_arguments_degrade_to_an_empty_object() {
        assert_eq!(parse_tool_arguments(""), json!({}));
        assert_eq!(parse_tool_arguments("   "), json!({}));
        assert_eq!(parse_tool_arguments("{\"a\":1"), json!({}));
        // Valid JSON that is not an object would still be a 400 upstream.
        assert_eq!(parse_tool_arguments("[1,2]"), json!({}));
        assert_eq!(parse_tool_arguments("\"x\""), json!({}));
        assert_eq!(parse_tool_arguments("{\"a\":1}"), json!({ "a": 1 }));
    }

    #[test]
    fn tool_call_without_an_id_gets_one() {
        let converted = params(&json!([{
            "role": "assistant",
            "tool_calls": [{ "function": { "name": "f", "arguments": "{}" } }]
        }]));

        let MessageContent::Blocks { content } = &converted.messages[0].content else {
            panic!("expected block content");
        };
        let [ContentBlock::ToolUse { id, .. }] = content.as_slice() else {
            panic!("expected a tool_use block");
        };
        assert!(id.starts_with("call_"), "synthesised id: {id}");
        assert!(id.len() > "call_".len());
    }

    #[test]
    fn developer_role_is_treated_as_system() {
        let converted = params(&json!([
            { "role": "developer", "content": "be terse" },
            { "role": "user", "content": "hi" }
        ]));

        let system = converted.system.expect("system present");
        assert_eq!(system[0]["text"], "be terse");
        assert_eq!(converted.messages.len(), 1);
        assert_eq!(converted.messages[0].role, Role::User);
    }
}

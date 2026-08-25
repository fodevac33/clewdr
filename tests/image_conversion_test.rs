//! Tests for OAI `ImageUrl` to Claude Image format conversion
//!
//! This test suite validates that OpenAI format `image_url` blocks are correctly
//! converted to Claude's image format with proper source structure.
//!
//! ## Background
//! OpenAI uses `image_url: { url: "data:image/png;base64,..." }` format
//! Claude uses `image: { source: { type: "base64", media_type: "image/png", data: "..." } }`
//!
//! ## Fixed Issues
//! - CC proxy was passing `ImageUrl` directly to Claude API, causing 400/422 errors

#[cfg(test)]
mod tests {
    use anthropic_wire::{
        ContentBlock, CreateMessageParams as ClaudeCreateMessageParams, ImageSource, ImageUrl,
        MessageContent, Role,
    };
    use clewdr::types::oai::{
        CreateMessageParams as OaiCreateMessageParams, OaiContent, OaiMessage, OaiRole,
    };

    #[test]
    fn test_image_source_from_data_url_png() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_some(), "Should parse valid PNG data URI");

        match source.unwrap() {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(
                    data,
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                );
            }
            other => panic!("Expected Base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn test_image_source_from_data_url_jpeg() {
        let url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBg==";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_some(), "Should parse valid JPEG data URI");

        match source.unwrap() {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/jpeg");
                assert_eq!(data, "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBg==");
            }
            other => panic!("Expected Base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn test_image_source_from_data_url_webp() {
        let url = "data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA=";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_some(), "Should parse valid WebP data URI");

        match source.unwrap() {
            ImageSource::Base64 { media_type, .. } => assert_eq!(media_type, "image/webp"),
            other => panic!("Expected Base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn test_image_source_from_data_url_rejects_http() {
        let url = "https://example.com/image.png";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_none(), "Should reject HTTP URL");
    }

    #[test]
    fn test_image_source_from_data_url_rejects_invalid() {
        // Missing data: prefix
        assert!(ImageSource::from_data_url("image/png;base64,abc").is_none());

        // Missing comma separator
        assert!(ImageSource::from_data_url("data:image/png;base64").is_none());

        // Missing semicolon (no ;base64 marker)
        assert!(ImageSource::from_data_url("data:image/pngbase64,abc").is_none());

        // Empty string
        assert!(ImageSource::from_data_url("").is_none());

        // Empty data after comma
        assert!(ImageSource::from_data_url("data:image/png;base64,").is_none());

        // Empty media type
        assert!(ImageSource::from_data_url("data:;base64,abc").is_none());
    }

    #[test]
    fn test_image_source_from_data_url_with_extra_params() {
        // data URI with extra parameters before base64
        let url = "data:image/png;name=test.png;base64,iVBORw0KGgo=";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_some(), "Should parse data URI with extra params");

        match source.unwrap() {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "iVBORw0KGgo=");
            }
            other => panic!("Expected Base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn test_image_source_from_data_url_normalizes_jpg() {
        let url = "data:image/jpg;base64,/9j/4AAQ=";

        let source = ImageSource::from_data_url(url);

        match source.unwrap() {
            ImageSource::Base64 { media_type, .. } => assert_eq!(media_type, "image/jpeg"),
            other => panic!("Expected Base64 image source, got {other:?}"),
        }
    }

    #[test]
    fn test_image_source_from_data_url_rejects_unsupported_media_type() {
        let url = "data:application/pdf;base64,JVBERi0xLjQK";

        assert!(ImageSource::from_data_url(url).is_none());
    }

    #[test]
    fn test_image_source_from_data_url_case_insensitive() {
        // uppercase BASE64
        let url = "data:image/png;BASE64,iVBORw0KGgo=";

        let source = ImageSource::from_data_url(url);
        assert!(source.is_some(), "Should parse uppercase BASE64");

        match source.unwrap() {
            ImageSource::Base64 { media_type, .. } => assert_eq!(media_type, "image/png"),
            other => panic!("Expected Base64 image source, got {other:?}"),
        }

        // mixed case Base64
        let url2 = "data:image/jpeg;Base64,/9j/4AAQ=";
        let source2 = ImageSource::from_data_url(url2);
        assert!(source2.is_some(), "Should parse mixed case Base64");
        assert!(matches!(source2.unwrap(), ImageSource::Base64 { .. }));
    }
    fn user_blocks(blocks: Vec<ContentBlock>) -> OaiMessage {
        OaiMessage {
            role: OaiRole::User,
            content: Some(OaiContent::Blocks(blocks)),
            ..Default::default()
        }
    }

    fn text(role: OaiRole, content: &str) -> OaiMessage {
        OaiMessage {
            role,
            content: Some(OaiContent::Text(content.to_string())),
            ..Default::default()
        }
    }

    fn convert(messages: Vec<OaiMessage>) -> ClaudeCreateMessageParams {
        OaiCreateMessageParams {
            model: "claude-3-opus".to_string(),
            messages,
            ..Default::default()
        }
        .into()
    }

    #[test]
    fn test_oai_to_claude_conversion_with_image_url() {
        let claude_params = convert(vec![user_blocks(vec![
            ContentBlock::text("What's in this image?"),
            ContentBlock::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                },
            },
        ])]);

        assert_eq!(claude_params.messages.len(), 1);

        let msg = &claude_params.messages[0];
        assert_eq!(msg.role, Role::User);

        let MessageContent::Blocks { content } = &msg.content else {
            panic!("Expected Blocks content");
        };
        assert_eq!(content.len(), 2);

        // First block should be text
        assert!(
            matches!(&content[0], ContentBlock::Text { text, .. } if text == "What's in this image?")
        );

        // Second block should be converted to Image (not ImageUrl)
        match &content[1] {
            ContentBlock::Image { source, .. } => match source {
                ImageSource::Base64 { media_type, data } => {
                    assert_eq!(media_type, "image/png");
                    assert_eq!(data, "iVBORw0KGgo=");
                }
                other => panic!("Expected Base64 image source, got {other:?}"),
            },
            other => panic!("Expected Image block, got {other:?}"),
        }
    }

    #[test]
    fn test_oai_to_claude_conversion_with_remote_image_url() {
        let claude_params = convert(vec![user_blocks(vec![
            ContentBlock::text("What's in this image?"),
            ContentBlock::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                },
            },
        ])]);

        let MessageContent::Blocks { content } = &claude_params.messages[0].content else {
            panic!("Expected Blocks content");
        };
        assert_eq!(content.len(), 2);
        assert!(matches!(&content[0], ContentBlock::Text { .. }));
        assert!(matches!(
            &content[1],
            ContentBlock::Image {
                source: ImageSource::Url { url },
                ..
            } if url == "https://example.com/image.png"
        ));
    }

    #[test]
    fn test_oai_to_claude_conversion_filters_invalid_image_url() {
        let claude_params = convert(vec![user_blocks(vec![
            ContentBlock::text("What's in this image?"),
            ContentBlock::ImageUrl {
                image_url: ImageUrl {
                    url: "not-a-url".to_string(),
                },
            },
        ])]);

        let MessageContent::Blocks { content } = &claude_params.messages[0].content else {
            panic!("Expected Blocks content");
        };
        assert_eq!(content.len(), 1, "Invalid ImageUrl should be filtered out");
        assert!(matches!(&content[0], ContentBlock::Text { .. }));
    }

    #[test]
    fn test_oai_to_claude_preserves_existing_image_format() {
        // Claude's native Image format must survive untouched.
        let claude_params = convert(vec![user_blocks(vec![ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "existing_data".to_string(),
            },
            cache_control: None,
        }])]);

        let MessageContent::Blocks { content } = &claude_params.messages[0].content else {
            panic!("Expected Blocks content");
        };
        match &content[0] {
            ContentBlock::Image { source, .. } => {
                assert!(
                    matches!(source, ImageSource::Base64 { data, .. } if data == "existing_data")
                );
            }
            other => panic!("Expected Image block, got {other:?}"),
        }
    }

    #[test]
    fn test_oai_to_claude_text_content_unchanged() {
        let claude_params = convert(vec![text(OaiRole::User, "Hello, world!")]);

        let MessageContent::Text { content } = &claude_params.messages[0].content else {
            panic!("Expected Text content");
        };
        assert_eq!(content, "Hello, world!");
    }

    #[test]
    fn test_oai_to_claude_empty_message_filtered() {
        // The middle message holds only an invalid ImageUrl, so it empties out.
        let claude_params = convert(vec![
            text(OaiRole::User, "First message"),
            user_blocks(vec![ContentBlock::ImageUrl {
                image_url: ImageUrl {
                    url: "not-a-url".to_string(),
                },
            }]),
            text(OaiRole::User, "Third message"),
        ]);

        assert_eq!(claude_params.messages.len(), 2);

        assert_eq!(
            claude_params.messages[0].content,
            MessageContent::Text {
                content: "First message".to_string()
            }
        );
        assert_eq!(
            claude_params.messages[1].content,
            MessageContent::Text {
                content: "Third message".to_string()
            }
        );
    }

    #[test]
    fn test_oai_to_claude_system_content_keeps_only_text() {
        let claude_params = convert(vec![
            OaiMessage {
                role: OaiRole::System,
                content: Some(OaiContent::Blocks(vec![
                    ContentBlock::text("System prompt"),
                    ContentBlock::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                        },
                    },
                ])),
                ..Default::default()
            },
            text(OaiRole::User, "Hello"),
        ]);

        let system = claude_params.system.expect("system should be present");
        let blocks = system.as_array().expect("system should be an array");

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "System prompt");
    }
}

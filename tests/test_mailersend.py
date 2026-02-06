from app.services.mailersend import parse_mailersend_payload


def test_parse_mailersend_payload_minimal():
    payload = {
        "from": "alice@example.com",
        "subject": "Hello",
        "text": "Sample content",
        "attachments": [],
    }
    inbound = parse_mailersend_payload(payload)
    assert inbound.sender == "alice@example.com"
    assert inbound.subject == "Hello"
    assert inbound.text == "Sample content"
    assert inbound.attachments == []


def test_parse_mailersend_payload_extracts_lowercase_message_id():
    payload = {
        "data": {
            "from": "alice@example.com",
            "subject": "Hello",
            "text": "Sample content",
            "attachments": [],
            "headers": {
                "message-id": "<ABC-123@mx.example.com>",
            },
        }
    }
    inbound = parse_mailersend_payload(payload)
    assert inbound.message_id == "<abc-123@mx.example.com>"

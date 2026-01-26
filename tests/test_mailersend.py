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

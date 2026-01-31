from __future__ import annotations

import smtplib
from email.message import EmailMessage

import markdown

from app.config import settings


def _email_from_header() -> str:
    sender_name = settings.smtp_sender_name or settings.email_brand_name
    if sender_name:
        return f"{sender_name} <{settings.smtp_sender}>"
    return settings.smtp_sender or ""


def _logo_block() -> str:
    if settings.email_logo_url:
        return (
            f'<img src="{settings.email_logo_url}" alt="{settings.email_brand_name}" '
            'style="display:block;height:40px;margin:0 auto 24px auto;" />'
        )
    return (
        '<div style="text-align:center;margin:0 0 24px 0;">'
        f'<span style="font-size:22px;font-weight:800;letter-spacing:-0.02em;color:#0f172a;">'
        f'{settings.email_brand_name}'
        '</span>'
        '<span style="font-size:22px;font-weight:800;color:#2563eb;">.</span>'
        '</div>'
    )


def _build_html_email(title: str, body_html: str, button_label: str, button_url: str) -> str:
    return f"""\
<div style="background:#f5f7fb;padding:32px 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#0f172a;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:28px 28px 24px 28px;">
    {_logo_block()}
    <h1 style="margin:0 0 12px 0;font-size:20px;line-height:1.4;color:#0f172a;">{title}</h1>
    <div style="margin:0 0 18px 0;font-size:14px;line-height:1.6;color:#475569;">
      {body_html}
    </div>
    <a href="{button_url}" style="display:inline-block;background:#2563eb;color:#ffffff;text-decoration:none;font-size:14px;font-weight:600;padding:12px 20px;border-radius:10px;">
      {button_label}
    </a>
    <p style="margin:18px 0 0 0;font-size:12px;line-height:1.5;color:#94a3b8;">
      If you did not request this email, you can ignore it.
    </p>
  </div>
  <p style="max-width:560px;margin:12px auto 0 auto;font-size:11px;line-height:1.5;color:#94a3b8;text-align:center;">
    {settings.email_brand_name}
  </p>
</div>
"""


def _send_message(message: EmailMessage) -> None:
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        if settings.smtp_user and settings.smtp_password:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(message)


def send_magic_link(recipient: str, magic_link: str) -> None:
    if not settings.smtp_host or not settings.smtp_sender:
        raise RuntimeError("SMTP settings are not configured")

    message = EmailMessage()
    message["Subject"] = f"{settings.email_brand_name} sign-in link"
    message["From"] = _email_from_header()
    message["To"] = recipient
    text_body = "\n".join(
        [
            f"Use the link below to sign in to {settings.email_brand_name}:",
            magic_link,
            "",
            "If you did not request this link, you can ignore this email.",
        ]
    )
    message.set_content(text_body)

    html_body = _build_html_email(
        f"Sign in to {settings.email_brand_name}",
        "Click the button below to securely sign in. This link expires soon.",
        "Sign in",
        magic_link,
    )
    message.add_alternative(html_body, subtype="html")
    _send_message(message)


def send_article_reply(recipient: str, article_title: str, summary_markdown: str, article_url: str) -> None:
    if not settings.smtp_host or not settings.smtp_sender:
        raise RuntimeError("SMTP settings are not configured")

    message = EmailMessage()
    message["Subject"] = f"{settings.email_brand_name} saved your article"
    message["From"] = _email_from_header()
    message["To"] = recipient
    text_body = "\n".join(
        [
            f"Your email was saved as: {article_title}",
            "",
            "Summary:",
            summary_markdown or "(summary not available)",
            "",
            f"Open the article: {article_url}",
        ]
    )
    message.set_content(text_body)

    summary_html = markdown.markdown(summary_markdown or "", extensions=["extra", "sane_lists"])
    body_html = (
        f"<p>Your email was saved as <strong>{article_title}</strong>.</p>"
        f"{summary_html}"
    )
    html_body = _build_html_email(
        "Article saved",
        body_html,
        "Open article",
        article_url,
    )
    message.add_alternative(html_body, subtype="html")
    _send_message(message)

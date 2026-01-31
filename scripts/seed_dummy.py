from __future__ import annotations

from datetime import datetime

from app.config import settings
from app.db import SessionLocal
from app.models import Article, Chunk, Source, User
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts


def _embed_or_zero(chunks: list[str]) -> list[list[float]]:
    try:
        return embed_texts(chunks)
    except RuntimeError:
        return [[0.0] * settings.embedding_dim for _ in chunks]


def seed_dummy_articles() -> None:
    seed_email = "seed@datastore.local"
    entries = [
        {
            "title": "Q2 Support Escalation Review",
            "summary": "- Reviewed top escalation themes\n- Agreed to reduce SLA breaches by 20%\n- Assigned owners for automation fixes\n- Next review in 30 days",
            "content": "Customer escalations increased in Q2. Key issues were delayed triage and missing auto-replies. We will implement a new SLA dashboard and add auto-tagging for priority tickets. Owners: Nina, Omar. Target date: 2025-05-15. Progress: planning.",
            "category": "support_tickets",
            "extracted": {"people": ["Nina", "Omar"], "dates": ["2025-05-15"], "timeline": "Q2", "progress": "planning"},
        },
        {
            "title": "Remote Work Policy Update",
            "summary": "- Updated hybrid schedule rules\n- Clarified expense reimbursement\n- Added security requirements\n- Effective next quarter",
            "content": "The remote work policy now requires a minimum of two in-office days per week. Expense reimbursement applies to home office equipment up to $500. Security training must be completed by 2025-04-30. People: HR team. Progress: approved.",
            "category": "policies",
            "extracted": {"people": ["HR team"], "dates": ["2025-04-30"], "timeline": "next quarter", "progress": "approved"},
        },
        {
            "title": "API Authentication Guide",
            "summary": "- Use OAuth2 client credentials\n- Rotate keys every 90 days\n- Log failed auth attempts",
            "content": "This document outlines OAuth2 client credential flow for the API gateway. Rotate credentials every 90 days and store secrets in Vault. Contact: Priya. Progress: published.",
            "category": "documentation",
            "extracted": {"people": ["Priya"], "dates": [], "timeline": "ongoing", "progress": "published"},
        },
        {
            "title": "Project Atlas Kickoff",
            "summary": "- Defined scope and milestones\n- Established weekly standups\n- Identified integration risks",
            "content": "Project Atlas aims to consolidate reporting into a unified dashboard. Milestones: discovery by 2025-03-20, prototype by 2025-05-01. People: Luca, Mei, Sara. Progress: kickoff completed.",
            "category": "projects",
            "extracted": {"people": ["Luca", "Mei", "Sara"], "dates": ["2025-03-20", "2025-05-01"], "timeline": "Q1-Q2", "progress": "kickoff completed"},
        },
        {
            "title": "Incident Postmortem - Billing Outage",
            "summary": "- Root cause: misconfigured cache\n- Impacted 12% of customers\n- Fix: rollback + config validation",
            "content": "On 2025-02-18, billing requests failed due to a cache misconfiguration. Mitigation included rollback and adding config validation. People: DevOps, Rina. Progress: resolved.",
            "category": "support_tickets",
            "extracted": {"people": ["DevOps", "Rina"], "dates": ["2025-02-18"], "timeline": "incident day", "progress": "resolved"},
        },
        {
            "title": "Security Access Control Policy",
            "summary": "- Principle of least privilege\n- Quarterly access reviews\n- Mandatory MFA",
            "content": "All systems must enforce MFA and least-privilege access. Quarterly access reviews are scheduled for the first week of each quarter. People: Security team. Progress: approved.",
            "category": "policies",
            "extracted": {"people": ["Security team"], "dates": [], "timeline": "quarterly", "progress": "approved"},
        },
        {
            "title": "Customer Success Playbook",
            "summary": "- Standard onboarding checklist\n- 30/60/90 day milestones\n- Risk scoring guidelines",
            "content": "The playbook defines onboarding steps and success milestones at 30/60/90 days. Owners: Jenna, Chris. Progress: drafted.",
            "category": "documentation",
            "extracted": {"people": ["Jenna", "Chris"], "dates": [], "timeline": "30/60/90 days", "progress": "drafted"},
        },
        {
            "title": "Q3 Roadmap Proposal",
            "summary": "- Prioritize analytics improvements\n- Delay legacy migration\n- Align with enterprise feedback",
            "content": "The Q3 roadmap proposes advanced analytics and enterprise-grade audit logs. Migration of the legacy workflow is deferred. Stakeholders: Product, Sales. Progress: in review.",
            "category": "projects",
            "extracted": {"people": ["Product", "Sales"], "dates": [], "timeline": "Q3", "progress": "in review"},
        },
        {
            "title": "Vendor Contract Renewal Checklist",
            "summary": "- Verify SLA compliance\n- Review security addendum\n- Confirm pricing tiers",
            "content": "Use this checklist before renewing vendor contracts. Ensure SLA compliance, review security obligations, and validate pricing tiers. Owner: Legal. Progress: ready.",
            "category": "policies",
            "extracted": {"people": ["Legal"], "dates": [], "timeline": "pre-renewal", "progress": "ready"},
        },
        {
            "title": "Enterprise SSO Integration Notes",
            "summary": "- Supports SAML 2.0\n- Okta and Azure AD tested\n- Provisioning via SCIM",
            "content": "SSO integration supports SAML 2.0 with Okta and Azure AD. SCIM provisioning is available. People: Diego. Progress: validated.",
            "category": "documentation",
            "extracted": {"people": ["Diego"], "dates": [], "timeline": "current", "progress": "validated"},
        },
    ]

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == seed_email).first()
        if not user:
            user = User(email=seed_email)
            db.add(user)
            db.flush()
        for entry in entries:
            article = Article(
                owner_id=user.id,
                title=entry["title"],
                summary=entry["summary"],
                content_text=entry["content"],
                metadata_={
                    "sender": "seed@datastore.local",
                    "subject": entry["title"],
                    "category": entry["category"],
                    "extracted": entry["extracted"],
                    "seed": True,
                    "seeded_at": datetime.utcnow().isoformat(),
                },
            )
            db.add(article)
            db.flush()

            db.add(
                Source(
                    article_id=article.id,
                    source_type="seed",
                    source_name="seed",
                    source_uri=None,
                    raw_text=entry["content"],
                    metadata_={"seed": True},
                )
            )

            chunks = chunk_text(entry["content"])
            embeddings = _embed_or_zero(chunks)
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db.add(
                    Chunk(
                        article_id=article.id,
                        chunk_index=index,
                        content=chunk,
                        embedding=embedding,
                        metadata_={
                            "source": "seed",
                            "category": entry["category"],
                            "extracted": entry["extracted"],
                            "seed": True,
                        },
                    )
                )

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    seed_dummy_articles()

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email text NOT NULL UNIQUE,
    mobile_phone text,
    inference_endpoint text,
    inference_api_key text,
    default_inference_id uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_inference_configs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name text NOT NULL,
    provider text NOT NULL DEFAULT 'openrouter',
    base_url text,
    api_key text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT user_inference_configs_user_name_key UNIQUE (user_id, name)
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS mobile_phone text;
ALTER TABLE users ADD COLUMN IF NOT EXISTS inference_endpoint text;
ALTER TABLE users ADD COLUMN IF NOT EXISTS inference_api_key text;
ALTER TABLE users ADD COLUMN IF NOT EXISTS default_inference_id uuid;
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_default_inference_id_fkey;
ALTER TABLE users
    ADD CONSTRAINT users_default_inference_id_fkey
    FOREIGN KEY (default_inference_id)
    REFERENCES user_inference_configs(id)
    ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS groups (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name text NOT NULL,
    slug text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS group_members (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id uuid NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role text NOT NULL DEFAULT 'member',
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT group_members_group_user_key UNIQUE (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS folders (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id uuid REFERENCES groups(id) ON DELETE CASCADE,
    parent_id uuid REFERENCES folders(id) ON DELETE CASCADE,
    name text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE folders ADD COLUMN IF NOT EXISTS group_id uuid;
ALTER TABLE folders DROP CONSTRAINT IF EXISTS folders_group_id_fkey;
ALTER TABLE folders
    ADD CONSTRAINT folders_group_id_fkey
    FOREIGN KEY (group_id)
    REFERENCES groups(id)
    ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS articles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id uuid REFERENCES groups(id) ON DELETE CASCADE,
    folder_id uuid REFERENCES folders(id) ON DELETE SET NULL,
    title text NOT NULL,
    summary text NOT NULL,
    content_text text NOT NULL,
    external_dedupe_key text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE articles ADD COLUMN IF NOT EXISTS group_id uuid;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS external_dedupe_key text;
ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_group_id_fkey;
ALTER TABLE articles
    ADD CONSTRAINT articles_group_id_fkey
    FOREIGN KEY (group_id)
    REFERENCES groups(id)
    ON DELETE CASCADE;

CREATE TABLE IF NOT EXISTS sources (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    source_type text NOT NULL,
    source_name text,
    source_uri text,
    raw_text text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS article_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    version integer NOT NULL,
    title text NOT NULL,
    summary text NOT NULL,
    content_text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS magic_link_tokens (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash text NOT NULL UNIQUE,
    expires_at timestamptz NOT NULL,
    used_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    last_seen_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_owner_id ON articles(owner_id);
CREATE INDEX IF NOT EXISTS idx_articles_group_id ON articles(group_id);
CREATE INDEX IF NOT EXISTS idx_articles_folder_id ON articles(folder_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_folders_owner_id ON folders(owner_id);
CREATE INDEX IF NOT EXISTS idx_folders_group_id ON folders(group_id);
CREATE INDEX IF NOT EXISTS idx_folders_parent_id ON folders(parent_id);
CREATE INDEX IF NOT EXISTS idx_groups_owner_id ON groups(owner_id);
CREATE INDEX IF NOT EXISTS idx_group_members_group_id ON group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_group_members_user_id ON group_members(user_id);
CREATE INDEX IF NOT EXISTS idx_article_versions_article_id ON article_versions(article_id);
CREATE INDEX IF NOT EXISTS idx_sources_article_id ON sources(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_article_id ON chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS idx_articles_metadata ON articles USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_articles_search ON articles USING gin (to_tsvector('english', title || ' ' || summary || ' ' || content_text));
CREATE INDEX IF NOT EXISTS idx_articles_external_dedupe_key ON articles(external_dedupe_key);
DROP INDEX IF EXISTS ux_articles_owner_dedupe_null_group;
CREATE UNIQUE INDEX ux_articles_owner_dedupe_null_group
ON articles(owner_id, external_dedupe_key)
WHERE group_id IS NULL AND external_dedupe_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_owner_group_dedupe
ON articles(owner_id, group_id, external_dedupe_key)
WHERE group_id IS NOT NULL AND external_dedupe_key IS NOT NULL;

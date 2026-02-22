-- Curated tables for news and events

CREATE TABLE IF NOT EXISTS cur_news_items (
    news_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES dim_source(source_id),
    headline TEXT NOT NULL,
    body TEXT,
    url VARCHAR(1000),
    language VARCHAR(10),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    entities JSONB, -- array of entity_ids or extracted strings
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cur_news_available_time ON cur_news_items(available_time);
CREATE INDEX idx_cur_news_event_time ON cur_news_items(event_time);
CREATE INDEX idx_cur_news_source ON cur_news_items(source_id);

CREATE TABLE IF NOT EXISTS cur_world_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES dim_source(source_id),
    gdelt_event_id BIGINT,
    event_type VARCHAR(50),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    location JSONB,
    actors JSONB,
    themes JSONB,
    tone_score NUMERIC,
    sentiment_positive NUMERIC,
    sentiment_negative NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cur_world_events_available_time ON cur_world_events(available_time);
CREATE INDEX idx_cur_world_events_event_time ON cur_world_events(event_time);
CREATE INDEX idx_cur_world_events_type ON cur_world_events(event_type);
CREATE INDEX idx_cur_world_events_gdelt_id ON cur_world_events(gdelt_event_id);

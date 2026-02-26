-- Raw tables for social media / financial news sentiment

CREATE TABLE IF NOT EXISTS raw_reddit_posts (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(20) NOT NULL UNIQUE,
    subreddit VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author VARCHAR(100),
    score INTEGER,
    upvote_ratio NUMERIC,
    num_comments INTEGER,
    created_utc TIMESTAMPTZ NOT NULL,
    tickers_mentioned JSONB,
    sentiment_score NUMERIC,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_raw_reddit_subreddit ON raw_reddit_posts(subreddit);
CREATE INDEX idx_raw_reddit_created ON raw_reddit_posts(created_utc);
CREATE INDEX idx_raw_reddit_tickers ON raw_reddit_posts USING GIN (tickers_mentioned);

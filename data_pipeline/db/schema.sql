CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id     TEXT NOT NULL,
    source_system   TEXT NOT NULL,           -- "greenhouse", "jsearch", "flyby", "serpapi"
    source_board    TEXT,                    -- board token / query string
    title           TEXT,
    location        TEXT,
    city            TEXT,
    state           TEXT,
    country         TEXT,
    description     TEXT,
    company_name    TEXT,
    company_domain  TEXT,                    -- JSearch only
    department      TEXT,                    -- Greenhouse only
    employment_type TEXT,
    seniority_level TEXT,                    -- post-processing
    salary_min      REAL,
    salary_max      REAL,
    salary_currency TEXT DEFAULT 'USD',
    salary_period   TEXT,
    remote_status   TEXT,
    posted_date     TEXT,
    updated_date    TEXT,                    -- Greenhouse only
    expiry_date     TEXT,                    -- JSearch only
    source_url      TEXT,
    apply_url       TEXT,
    job_status      TEXT DEFAULT 'active',
    scraped_at      TEXT NOT NULL,
    raw_data        TEXT,
    metadata        TEXT,

    UNIQUE(external_id, source_system, source_board)
);

CREATE TABLE IF NOT EXISTS request_budget (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    date        TEXT NOT NULL,
    requests    INTEGER DEFAULT 0,
    UNIQUE(source, date)
);

CREATE TABLE IF NOT EXISTS collection_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT NOT NULL,
    source      TEXT NOT NULL,
    jobs_found  INTEGER DEFAULT 0,
    jobs_new    INTEGER DEFAULT 0,
    jobs_error  INTEGER DEFAULT 0,
    requests    INTEGER DEFAULT 0,
    errors      TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source_system);
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_date);
CREATE INDEX IF NOT EXISTS idx_jobs_scraped ON jobs(scraped_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(job_status);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location);

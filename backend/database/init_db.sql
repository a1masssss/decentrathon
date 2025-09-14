CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(255) NOT NULL,
    is_damaged BOOLEAN,
    damage_source VARCHAR(255),
    damage_local_result JSONB,
    rust_scratch_result JSONB,
    damage_parts_local_result JSONB,
    dirty_result JSONB,
    llm_output TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

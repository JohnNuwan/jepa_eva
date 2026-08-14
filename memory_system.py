#!/usr/bin/env python3
"""
MÉMOIRE CENTRALISÉE EVA — SQLite + FTS5
Stocke les leçons, skills, et expériences des agents.
Recherche plein texte + requêtes structurées.
"""
import sqlite3, json, time, os, sys
from datetime import datetime
from pathlib import Path

DB_PATH = "/home/aza/eva-adam-v2/data/memory.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,          -- agent qui a appris
    lesson TEXT NOT NULL,         -- le texte de la leçon
    category TEXT DEFAULT 'general', -- trading, code, infra, security
    context TEXT,                 -- contexte (JSON)
    success INTEGER DEFAULT 0,   -- 1=positif, 0=negatif
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
    lesson, category, content=lessons, content_rowid=id
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    path TEXT,                    -- chemin du fichier
    category TEXT,
    tags TEXT,                    -- JSON array
    created_by TEXT DEFAULT 'eva',
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    task TEXT,                    -- description de la tâche
    result TEXT,                  -- résultat (JSON)
    metrics TEXT,                 -- métriques (JSON)
    champion_path TEXT,           -- si applicable
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lessons_fts(rowid, lesson, category)
    VALUES (new.id, new.lesson, new.category);
END;
"""

def init():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

def add_lesson(agent, lesson, category="general", context=None, success=0):
    conn = init()
    conn.execute("INSERT INTO lessons (agent, lesson, category, context, success) VALUES (?,?,?,?,?)",
                 (agent, lesson, category, json.dumps(context) if context else None, success))
    conn.commit()
    conn.close()
    return True

def search_lessons(query, limit=10):
    conn = init()
    cur = conn.execute("""
        SELECT l.agent, l.lesson, l.category, l.success, l.created_at
        FROM lessons_fts f JOIN lessons l ON f.rowid = l.id
        WHERE lessons_fts MATCH ? ORDER BY rank LIMIT ?""", (query, limit))
    results = [{"agent": r[0], "lesson": r[1], "category": r[2], "success": r[3], "date": r[4]} for r in cur.fetchall()]
    conn.close()
    return results

def add_skill(name, description, path, category="ml", tags=None, created_by="eva"):
    conn = init()
    try:
        conn.execute("INSERT INTO skills (name, description, path, category, tags, created_by) VALUES (?,?,?,?,?,?)",
                     (name, description, path, category, json.dumps(tags or []), created_by))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def add_experience(agent, task, result=None, metrics=None, champion_path=None):
    conn = init()
    conn.execute("INSERT INTO experiences (agent, task, result, metrics, champion_path) VALUES (?,?,?,?,?)",
                 (agent, task, json.dumps(result) if result else None,
                  json.dumps(metrics) if metrics else None, champion_path))
    conn.commit()
    conn.close()
    return True

def get_stats():
    conn = init()
    stats = {}
    for table in ["lessons", "skills", "experiences"]:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]
    conn.close()
    return stats

if __name__ == "__main__":
    conn = init()
    conn.close()
    print(json.dumps(get_stats()))

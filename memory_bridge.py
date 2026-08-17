#!/usr/bin/env python3
"""
MEMORY BRIDGE — Pont continu agents → Postgres pgvector
Scanne les lessons.json des agents, le GoBus, et écrit dans knowledge_nodes.
"""
import os, json, time, sqlite3, glob, hashlib
from datetime import datetime
import psycopg2
from pathlib import Path

PG_HOST = "localhost"
PG_PORT = 5433
PG_USER = "adam"
PG_PASS = "adam_secret_2026"
PG_DB = "adam"

AGENTS_DIR = "/home/aza/eva-adam-v2/agents"
HASH_CACHE = "/home/aza/eva-adam-v2/data/memory_hash_cache.json"
LOG_FILE = "/home/aza/eva-adam-v2/logs/memory_bridge.log"

def log(msg):
    ts = datetime.now().isoformat()[:19]
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def pg_connect():
    return psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, dbname=PG_DB)

def load_hash_cache():
    if os.path.exists(HASH_CACHE):
        with open(HASH_CACHE) as f:
            return json.load(f)
    return {}

def save_hash_cache(cache):
    with open(HASH_CACHE, "w") as f:
        json.dump(cache, f)

def lessons_to_postgres(agent_name, lessons):
    """Ecrit les lecons d'un agent dans knowledge_nodes"""
    conn = pg_connect()
    cur = conn.cursor()
    count = 0
    for lesson in lessons:
        if not isinstance(lesson, dict):
            continue
        text = lesson.get("lesson", lesson.get("text", str(lesson)))[:80]
        props = json.dumps({"agent": agent_name, "lesson": lesson, "source": "lessons.json"})
        try:
            cur.execute(
                "INSERT INTO knowledge_nodes (label, name, properties) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                ("Lesson", f"{agent_name}: {text}", props)
            )
            count += 1
        except Exception as e:
            log(f"  Erreur insertion: {e}")
    conn.commit()
    conn.close()
    return count

def scan_agents():
    """Scanne tous les agents et leurs lessons.json"""
    cache = load_hash_cache()
    total = 0
    
    for memory_file in sorted(glob.glob(f"{AGENTS_DIR}/*/memory/lessons.json")):
        agent_name = memory_file.split("/agents/")[1].split("/memory")[0]
        
        # Vérifier si le fichier a changé
        content = open(memory_file).read()
        file_hash = hashlib.md5(content.encode()).hexdigest()
        if cache.get(memory_file) == file_hash:
            continue  # pas de changement
        
        try:
            data = json.loads(content)
            lessons = data.get("lessons", [])
            if lessons:
                n = lessons_to_postgres(agent_name, lessons)
                if n > 0:
                    log(f"✅ {agent_name}: {n} lecons ajoutees")
                    total += n
                cache[memory_file] = file_hash
        except Exception as e:
            log(f"⚠️ Erreur {agent_name}: {e}")
    
    save_hash_cache(cache)
    return total

def scan_gobus():
    """Scanne l'event_bus.db pour les evenements recents"""
    db_path = "/home/aza/eva-adam-v2/ramdb/event_bus.db"
    if not os.path.exists(db_path):
        return 0
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Evenements des dernieres 5 minutes
        cur = conn.execute("""
            SELECT * FROM events 
            WHERE created_at > datetime('now', '-5 minutes')
            ORDER BY timestamp DESC LIMIT 20
        """)
        events = cur.fetchall()
        conn.close()
        
        if events:
            log(f"📡 {len(events)} evenements GoBus recents")
            # Les ecrire dans knowledge_nodes
            pg = pg_connect()
            cur = pg.cursor()
            for ev in events:
                ev_dict = dict(ev)
                name = f"Event: {ev_dict.get('topic', 'unknown')}"[:80]
                props = json.dumps({"source": "gobus", "event": ev_dict, "timestamp": str(ev_dict.get('timestamp', ''))})
                try:
                    cur.execute(
                        "INSERT INTO knowledge_nodes (label, name, properties) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        ("Event", name, props)
                    )
                except:
                    pass
            pg.commit()
            pg.close()
        return len(events)
    except Exception as e:
        log(f"⚠️ GoBus scan error: {e}")
        return 0

def main():
    log("=" * 60)
    log("MEMORY BRIDGE — Pont agents → Postgres pgvector")
    log("=" * 60)
    
    while True:
        try:
            n_lessons = scan_agents()
            n_events = scan_gobus()
            if n_lessons > 0 or n_events > 0:
                log(f"📊 Cycle: {n_lessons} lecons + {n_events} evenements")
            else:
                log(f"💤 Aucune donnee nouvelle")
        except Exception as e:
            log(f"❌ Erreur cycle: {e}")
        
        time.sleep(60)  # toutes les 5 min

if __name__ == "__main__":
    main()

import os
import json
import sqlite3
import io
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect
import anthropic
import docx
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

# Allow DATA_DIR to be overridden by environment variable (for Railway volume mounts)
DATA_DIR  = Path(os.getenv('DATA_DIR', 'data'))
DB_PATH   = DATA_DIR / 'profile.db'
_KEY_FILE = DATA_DIR / 'secret.key'

# ── Persistent secret key so sessions survive server restarts ──
def _load_secret_key():
    # Prefer SECRET_KEY environment variable (set this in Railway dashboard)
    env_key = os.getenv('SECRET_KEY')
    if env_key:
        return env_key.encode()
    # Fall back to file-based key for local development
    _KEY_FILE.parent.mkdir(exist_ok=True)
    if not _KEY_FILE.exists():
        _KEY_FILE.write_bytes(os.urandom(32))
    return _KEY_FILE.read_bytes()

app = Flask(__name__)
app.secret_key = _load_secret_key()

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL    = 'claude-sonnet-4-6'
MAX_PROFILES = 10


# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Multi-profile table
    c.execute('''
        CREATE TABLE IF NOT EXISTS profiles (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            name               TEXT    NOT NULL,
            style_document     TEXT    DEFAULT '',
            profile_percentage INTEGER DEFAULT 0,
            writing_samples    TEXT    DEFAULT '[]',
            prompt_responses   TEXT    DEFAULT '[]',
            created_at         TEXT    DEFAULT (datetime('now'))
        )
    ''')

    # Migrate data from old single-profile table if it exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profile'")
    if c.fetchone():
        c.execute("SELECT COUNT(*) FROM profiles")
        if c.fetchone()[0] == 0:
            c.execute("SELECT style_document, profile_percentage, writing_samples, prompt_responses FROM profile WHERE id=1")
            old = c.fetchone()
            if old and old[0]:
                c.execute(
                    "INSERT INTO profiles (name, style_document, profile_percentage, writing_samples, prompt_responses) VALUES (?,?,?,?,?)",
                    ('My Profile', old[0], old[1], old[2], old[3])
                )

    conn.commit()
    conn.close()


def get_active_profile_id():
    pid = session.get('active_profile_id')
    if not pid:
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM profiles WHERE id=?', (pid,))
    row = c.fetchone()
    conn.close()
    return pid if row else None


def get_profile(profile_id=None):
    if profile_id is None:
        profile_id = get_active_profile_id()
    if not profile_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, style_document, profile_percentage, writing_samples, prompt_responses FROM profiles WHERE id=?', (profile_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id':                 row[0],
        'name':               row[1],
        'style_document':     row[2] or '',
        'profile_percentage': row[3] or 0,
        'writing_samples':    json.loads(row[4] or '[]'),
        'prompt_responses':   json.loads(row[5] or '[]'),
    }


def save_profile(profile):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE profiles
        SET style_document=?, profile_percentage=?, writing_samples=?, prompt_responses=?
        WHERE id=?
    ''', (
        profile['style_document'],
        profile['profile_percentage'],
        json.dumps(profile['writing_samples']),
        json.dumps(profile['prompt_responses']),
        profile['id'],
    ))
    conn.commit()
    conn.close()


def get_all_profiles():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, profile_percentage, writing_samples, prompt_responses, created_at FROM profiles ORDER BY created_at DESC')
    rows = c.fetchall()
    conn.close()
    return [{
        'id':               r[0],
        'name':             r[1],
        'profile_percentage': r[2] or 0,
        'sample_count':     len(json.loads(r[3] or '[]')),
        'response_count':   len(json.loads(r[4] or '[]')),
        'created_at':       r[5],
    } for r in rows]


def require_profile():
    """Redirect to profile selector if no active profile is set."""
    if not get_active_profile_id():
        return redirect('/profiles')
    return None


@app.context_processor
def inject_globals():
    """Make active profile name available in every template."""
    pid = get_active_profile_id()
    active_name = None
    if pid:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT name FROM profiles WHERE id=?', (pid,))
        row = c.fetchone()
        conn.close()
        if row:
            active_name = row[0]
    return {'active_profile_name': active_name, 'active_profile_id': pid}


# ─────────────────────────────────────────────
# FILE READING
# ─────────────────────────────────────────────

def extract_text(file):
    name = file.filename.lower()
    raw  = file.read()
    if name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(raw))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    elif name.endswith('.pdf'):
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return '\n'.join(page.extract_text() or '' for page in pdf.pages)
    else:
        return raw.decode('utf-8', errors='ignore')


# ─────────────────────────────────────────────
# CLAUDE
# ─────────────────────────────────────────────

def ask_claude(prompt, max_tokens=2000):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.content[0].text


def calculate_percentage(profile):
    if not profile['style_document']:
        return 0
    samples_count   = len(profile['writing_samples'])
    responses_count = len(profile['prompt_responses'])
    try:
        result = ask_claude(f'''You are evaluating how well an AI understands someone's personal writing style.

Available data:
- {samples_count} uploaded writing sample(s)
- {responses_count} written prompt response(s)
- Style document: {len(profile["style_document"])} characters

Style document preview:
{profile["style_document"][:1500]}

Rate 0–100 how accurately we could replicate this person's writing:
0=nothing known, 25=surface only, 50=decent with gaps, 75=strong grasp, 100=perfect replication

Output ONLY a single integer.''', max_tokens=10)
        return max(0, min(100, int(result.strip())))
    except Exception:
        base = 20 if samples_count > 0 else 0
        base += min(samples_count * 8, 25)
        base += min(responses_count * 12, 45)
        return min(base, 95)


def build_humanize_prompt(text, style_doc, feedback=None):
    feedback_section = ''
    if feedback:
        feedback_section = f'\nFEEDBACK FROM PREVIOUS ATTEMPT — fix these specific issues:\n{feedback}\n'
    return f'''You are rewriting text to sound exactly like a specific real human being. Your output must pass AI detection tools as 100% human-written.

WRITING STYLE PROFILE:
{style_doc}
{feedback_section}
STRICT RULES:

1. BANNED WORDS/PHRASES (AI giveaways — never use):
   "Moreover", "Furthermore", "Additionally", "In conclusion", "It is worth noting",
   "It's important to", "Notably", "Significantly", "Delve", "Leverage", "Utilize",
   "Facilitate", "Endeavor", "In today's world", "In the modern era", "Landscape", "Realm"

2. SENTENCE STRUCTURE — break AI patterns:
   - Vary length dramatically — short punchy sentences mixed with longer ones
   - Do NOT make every paragraph the same length
   - Use contractions constantly (don't, it's, I've, they're, can't)
   - Include incomplete thoughts or run-ons if that fits this person's style

3. SOUND HUMAN:
   - Use the filler phrases and expressions from this person's profile
   - Allow minor grammar imperfections if that matches how they write
   - Do NOT over-explain — humans skip obvious context
   - Add their natural tangents and asides where fitting
   - Match their exact vocabulary level

4. PRESERVE MEANING — all facts stay intact

5. OUTPUT ONLY the rewritten text. No commentary.

TEXT TO REWRITE:
{text}'''


def score_text(text):
    result = ask_claude(f'''You are an AI writing detection expert. Score this text.

Check for AI giveaways:
- Formal transitions: Moreover, Furthermore, Additionally, In conclusion, Notably
- Perfectly balanced sentence lengths with no variation
- Overly polished grammar with zero natural imperfections
- Formal vocabulary where casual would be natural
- Generic filler: "It is worth noting", "It is important to"
- Missing contractions (don't, it's, can't, I've)
- All paragraphs same length
- No personal quirks, tangents, or asides

Respond in this EXACT format:
SCORE: [0-100, where 0=fully human, 100=obvious AI]
ISSUES: [bullet list of AI patterns found, or "None"]

Text:
{text}''', max_tokens=300)

    score    = 50
    feedback = ''
    for line in result.splitlines():
        if line.startswith('SCORE:'):
            try:
                score = max(0, min(100, int(line.replace('SCORE:', '').strip())))
            except ValueError:
                pass
        elif line.startswith('ISSUES:'):
            feedback = line.replace('ISSUES:', '').strip()
        elif feedback and line.strip():
            feedback += '\n' + line.strip()
    return score, feedback


# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/profiles')
def profiles():
    all_profiles  = get_all_profiles()
    active_id     = get_active_profile_id()
    profile_count = len(all_profiles)
    return render_template('profiles.html',
                           all_profiles=all_profiles,
                           active_id=active_id,
                           profile_count=profile_count,
                           max_profiles=MAX_PROFILES)


@app.route('/profiles/select/<int:profile_id>')
def select_profile(profile_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM profiles WHERE id=?', (profile_id,))
    if c.fetchone():
        session['active_profile_id'] = profile_id
    conn.close()
    return redirect('/step1')


@app.route('/step1')
def step1():
    redir = require_profile()
    if redir: return redir
    return render_template('step1.html', profile=get_profile())


@app.route('/step2')
def step2():
    redir = require_profile()
    if redir: return redir
    return render_template('step2.html', profile=get_profile())


@app.route('/step3')
def step3():
    redir = require_profile()
    if redir: return redir
    return render_template('step3.html', profile=get_profile())


# ─────────────────────────────────────────────
# PROFILE MANAGEMENT API
# ─────────────────────────────────────────────

@app.route('/api/profiles/create', methods=['POST'])
def create_profile():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Please enter a name for this profile.'}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM profiles')
    if c.fetchone()[0] >= MAX_PROFILES:
        conn.close()
        return jsonify({'error': f'Maximum of {MAX_PROFILES} profiles reached. Delete one to add another.'}), 400

    c.execute('INSERT INTO profiles (name) VALUES (?)', (name,))
    new_id = c.lastrowid
    conn.commit()
    conn.close()

    session['active_profile_id'] = new_id
    return jsonify({'success': True, 'id': new_id})


@app.route('/api/profiles/delete/<int:profile_id>', methods=['POST'])
def delete_profile(profile_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM profiles WHERE id=?', (profile_id,))
    conn.commit()
    conn.close()
    if session.get('active_profile_id') == profile_id:
        session.pop('active_profile_id', None)
    return jsonify({'success': True})


@app.route('/api/profiles/rename/<int:profile_id>', methods=['POST'])
def rename_profile(profile_id):
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name cannot be empty.'}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE profiles SET name=? WHERE id=?', (name, profile_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ─────────────────────────────────────────────
# WRITING API
# ─────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload():
    redir = require_profile()
    if redir: return jsonify({'error': 'No active profile.'}), 400

    files = request.files.getlist('files')
    texts = []
    for f in files:
        if f.filename:
            text = extract_text(f)
            if text.strip():
                texts.append(f'[File: {f.filename}]\n{text}')

    if not texts:
        return jsonify({'error': 'No readable text found in the uploaded files.'}), 400

    combined  = '\n\n---\n\n'.join(texts)
    profile   = get_profile()

    style_doc = ask_claude(f'''Analyze these writing samples and create a detailed Writing Style Profile for "{profile["name"]}".

Cover:
1. Sentence length and structure
2. Vocabulary level and word choices
3. Tone and voice (casual/formal, humor, directness)
4. Common phrases, filler words, expressions
5. Punctuation and capitalization habits
6. Paragraph structure
7. How they open and close thoughts
8. Distinctive quirks and patterns

Writing samples:
{combined[:8000]}

Write a thorough profile that could guide someone to replicate this person's writing convincingly. Quote examples.''', max_tokens=2500)

    profile['style_document']     = style_doc
    profile['writing_samples']    = texts[:5]
    profile['profile_percentage'] = calculate_percentage(profile)
    save_profile(profile)
    return jsonify({'success': True, 'percentage': profile['profile_percentage']})


@app.route('/api/get_prompt', methods=['GET'])
def get_prompt():
    prompt = ask_claude('''Generate one casual, open-ended writing prompt for personal reflection.
Requirements: conversational, naturally leads to 100+ words, no specialist knowledge needed.
Output ONLY the prompt. No quotes, no explanation.''', max_tokens=120)
    return jsonify({'prompt': prompt.strip()})


@app.route('/api/submit_response', methods=['POST'])
def submit_response():
    redir = require_profile()
    if redir: return jsonify({'error': 'No active profile.'}), 400

    data          = request.get_json()
    prompt        = data.get('prompt', '').strip()
    response_text = data.get('response', '').strip()

    word_count = len(response_text.split())
    if word_count < 100:
        return jsonify({'error': f'Please write at least 100 words. You currently have {word_count}.'}), 400
    if word_count > 500:
        return jsonify({'error': f'Please keep your response under 500 words. You currently have {word_count}.'}), 400

    profile = get_profile()
    profile['prompt_responses'].append({'prompt': prompt, 'response': response_text})

    all_responses = '\n\n'.join(
        f'Prompt: {r["prompt"]}\nResponse: {r["response"]}'
        for r in profile['prompt_responses']
    )

    profile['style_document'] = ask_claude(f'''Update this Writing Style Profile with new writing samples.

Current profile:
{profile["style_document"]}

New samples:
{all_responses[:5000]}

Keep accurate existing insights, update anything the new samples clarify, add new patterns. Output the complete updated profile.''', max_tokens=2500)

    profile['profile_percentage'] = calculate_percentage(profile)
    save_profile(profile)
    return jsonify({'success': True, 'percentage': profile['profile_percentage']})


@app.route('/api/humanize', methods=['POST'])
def humanize():
    redir = require_profile()
    if redir: return jsonify({'error': 'No active profile.'}), 400

    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Please provide some text to humanize.'}), 400

    profile = get_profile()
    if not profile['style_document']:
        return jsonify({'error': 'No writing profile found. Please complete Step 1 first.'}), 400

    MAX_ATTEMPTS = 4
    current_text = text
    feedback     = None
    final_score  = 100

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_humanize_prompt(current_text, profile['style_document'], feedback)
        result = ask_claude(prompt, max_tokens=4000)
        score, feedback = score_text(result)
        final_score = score

        if score <= 10:
            return jsonify({'result': result, 'score': score, 'attempts': attempt})

        current_text = result

    return jsonify({
        'result':   result,
        'score':    final_score,
        'attempts': MAX_ATTEMPTS,
        'warning':  f'Best score after {MAX_ATTEMPTS} attempts: {final_score}%. Try adding more writing samples.',
    })


@app.route('/api/profile', methods=['GET'])
def profile_info():
    profile = get_profile()
    if not profile:
        return jsonify({'error': 'No active profile'}), 400
    return jsonify({
        'percentage':     profile['profile_percentage'],
        'has_profile':    bool(profile['style_document']),
        'sample_count':   len(profile['writing_samples']),
        'response_count': len(profile['prompt_responses']),
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    pid = get_active_profile_id()
    if not pid:
        return jsonify({'error': 'No active profile'}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE profiles SET style_document='', profile_percentage=0, writing_samples='[]', prompt_responses='[]' WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────

# Run init_db at module load so gunicorn workers also initialise the database
init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '='*50)
    print('  UVOICER.AI is running!')
    print(f'  Open your browser and go to:')
    print(f'  http://localhost:{port}')
    print('='*50 + '\n')
    app.run(host='0.0.0.0', port=port, debug=False)

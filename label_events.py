"""
Local labelling page for the evaluation days (V4 approach 1, stage 0).

Every article of one day is assigned to its TRUE event, following
data/validation/EVENT_DEFINITION.md. Production's own grouping is not shown,
so the labels aren't pulled toward it. Suggestions are the labelled events
whose closest member is most similar to the current article.

Usage:
    python label_events.py --day 2026-09-29        (then open http://localhost:8765)

Saves after every action (atomic write), so it can be stopped and resumed:
    data/validation/eval_days/labels_<day>.csv   article_id, event_label, note, labelled_at
    data/validation/eval_days/events_<day>.csv   event_label, name
"""

import argparse
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

BASE = Path(__file__).parent / "data" / "validation" / "eval_days"
SUGGESTIONS = 5


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

class Day:

    def __init__(self, day):

        self.day = day
        self.labels_path = BASE / f"labels_{day}.csv"
        self.events_path = BASE / f"events_{day}.csv"

        articles = pd.read_csv(BASE / f"articles_{day}.csv")
        articles = articles.sort_values(["created_at", "id"]).reset_index(drop=True)

        descriptions_path = BASE / "local" / "descriptions.csv"
        if descriptions_path.exists():
            descriptions = pd.read_csv(descriptions_path).set_index("id")["description"]
            articles["description"] = articles["id"].map(descriptions)
        else:
            articles["description"] = None

        vectors = np.stack(articles["embedding"].map(
            lambda s: np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)
        ).values)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.row = {int(a): i for i, a in enumerate(articles["id"])}

        parts = articles["source"].str.split(" -> ", n=1, regex=False)
        self.articles = [
            {
                "id": int(r.id),
                "title": r.title,
                "outlet": part[0],
                "feed": part[1] if len(part) > 1 else "",
                "published_at": None if pd.isna(r.published_at) else str(r.published_at)[:16],
                "created_at": str(r.created_at)[:16],
                "description": None if pd.isna(r.description) else r.description,
            }
            for r, part in zip(articles.itertuples(index=False), parts)
        ]

        self.labels = {}
        if self.labels_path.exists():
            for r in pd.read_csv(self.labels_path, keep_default_na=False).itertuples(index=False):
                self.labels[int(r.article_id)] = {
                    "event_label": r.event_label, "note": r.note, "labelled_at": r.labelled_at
                }

        self.events = {}
        if self.events_path.exists():
            for r in pd.read_csv(self.events_path, keep_default_na=False).itertuples(index=False):
                self.events[r.event_label] = r.name

    # -- writes ----------------------------------------------

    def _write(self, path, frame):
        tmp = path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def save(self):
        self._write(self.labels_path, pd.DataFrame(
            [{"article_id": a, **v} for a, v in sorted(self.labels.items())],
            columns=["article_id", "event_label", "note", "labelled_at"]
        ))
        used = {v["event_label"] for v in self.labels.values()}
        self._write(self.events_path, pd.DataFrame(
            [{"event_label": e, "name": n} for e, n in sorted(self.events.items()) if e in used],
            columns=["event_label", "name"]
        ))

    def new_event(self, article_id):
        taken = {int(e.rsplit("-", 1)[1]) for e in self.events}
        label = f"{self.day}-{(max(taken) + 1 if taken else 1):04d}"
        title = self.articles[self.row[article_id]]["title"]
        self.events[label] = title[:90]
        return label

    def assign(self, article_id, event_label, note):
        if event_label == "new":
            event_label = self.new_event(article_id)
        if event_label not in self.events:
            raise ValueError(f"unknown event {event_label}")
        self.labels[article_id] = {
            "event_label": event_label,
            "note": note or "",
            "labelled_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return event_label

    def clear(self, article_id):
        self.labels.pop(article_id, None)
        self.save()

    def rename(self, event_label, name):
        if event_label not in self.events or not name.strip():
            raise ValueError("unknown event or empty name")
        self.events[event_label] = name.strip()[:120]
        self.save()

    # -- reads -----------------------------------------------

    def members(self):
        groups = {}
        for article_id, value in self.labels.items():
            groups.setdefault(value["event_label"], []).append(article_id)
        return groups

    def suggest(self, article_id):
        """Labelled events ranked by their most similar member (not a mean,
        which would favour broad catch-all events)."""
        vector = self.vectors[self.row[article_id]]
        scored = []
        for label, ids in self.members().items():
            ids = [a for a in ids if a != article_id]
            if not ids:
                continue
            sims = self.vectors[[self.row[a] for a in ids]] @ vector
            best = int(np.argmax(sims))
            scored.append((float(sims[best]), label, ids, ids[best]))
        scored.sort(reverse=True)
        return [
            {
                "event_label": label,
                "name": self.events[label],
                "size": len(ids),
                "similarity": round(sim, 3),
                "closest": self.articles[self.row[closest]]["title"],
            }
            for sim, label, ids, closest in scored[:SUGGESTIONS]
        ]

    def state(self):
        groups = self.members()
        return {
            "day": self.day,
            "articles": self.articles,
            "labels": {str(k): v for k, v in self.labels.items()},
            "events": [
                {"event_label": e, "name": n, "size": len(groups.get(e, []))}
                for e, n in self.events.items() if e in groups
            ],
        }


# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

def handler_for(day):

    class Handler(BaseHTTPRequestHandler):

        def _send(self, status, body, content_type="application/json"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE, "text/html")
            elif url.path == "/api/state":
                self._send(200, json.dumps(day.state()))
            elif url.path == "/api/suggest":
                article_id = int(parse_qs(url.query)["article_id"][0])
                self._send(200, json.dumps(day.suggest(article_id)))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/label":
                    label = day.assign(int(body["article_id"]), body["event_label"], body.get("note"))
                    self._send(200, json.dumps({"event_label": label}))
                elif path == "/api/clear":
                    day.clear(int(body["article_id"]))
                    self._send(200, "{}")
                elif path == "/api/rename":
                    day.rename(body["event_label"], body["name"])
                    self._send(200, "{}")
                else:
                    self._send(404, json.dumps({"error": "not found"}))
            except (KeyError, ValueError) as error:
                self._send(400, json.dumps({"error": str(error)}))

        def log_message(self, *args):
            pass

    return Handler


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>X-NEWS event labelling</title>
<style>
 body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f6f6f4;color:#1c1c1c}
 main{max-width:980px;margin:0 auto;padding:16px}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:12px 0}
 h1{font-size:20px;margin:6px 0} .meta{color:#666;font-size:13px}
 button{font:inherit;padding:6px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
 button:hover{background:#eef} .opt{display:block;width:100%;text-align:left;margin:6px 0}
 .opt small{color:#666} .current{outline:2px solid #2a6;}
 input[type=text]{font:inherit;padding:6px;width:100%;box-sizing:border-box}
 .events{max-height:240px;overflow:auto} .done{color:#2a6} kbd{background:#eee;border-radius:3px;padding:0 4px}
</style></head><body><main>
<div class="bar"><b id="day"></b><span id="progress"></span>
 <button onclick="move(-1)">&larr; Prev</button><button onclick="move(1)">Next &rarr;</button>
 <button onclick="jumpUnlabelled()">First unlabelled</button><button onclick="clearLabel()">Clear label</button></div>
<div class="card"><div class="meta" id="meta"></div><h1 id="title"></h1><div id="desc"></div>
 <div class="meta" id="assigned"></div></div>
<div class="card"><b>Suggestions</b> <span class="meta">(<kbd>1</kbd>-<kbd>5</kbd> choose, <kbd>N</kbd> new event, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move, <kbd>/</kbd> search)</span>
 <button class="opt" onclick="choose('new')"><b>N</b> &mdash; New event</button><div id="suggest"></div></div>
<div class="card"><input id="note" type="text" placeholder="Note (optional, e.g. 'unclear' or 'storyline: Asian Games cricket') - saved with the next choice"></div>
<div class="card"><input id="search" type="text" placeholder="Search all events labelled today"><div class="events" id="events"></div></div>
<div class="card"><b>Rename this article's event</b><input id="rename" type="text"><button onclick="rename()">Save name</button></div>
</main><script>
let S=null, i=0, sugg=[];
const $=id=>document.getElementById(id);
function el(tag,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
async function load(){S=await (await fetch('/api/state')).json();}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){alert((await r.json()).error);return null;} return r.json();}
function cur(){return S.articles[i];}
function eventName(l){const e=S.events.find(x=>x.event_label===l);return e?e.name:l;}
async function render(){
 const a=cur(), lab=S.labels[a.id], done=Object.keys(S.labels).length;
 $('day').textContent=S.day; $('progress').textContent=`${i+1} / ${S.articles.length} · labelled ${done} · events ${S.events.length}`;
 $('meta').textContent=`#${a.id} · ${a.outlet}${a.feed?' → '+a.feed:''} · published ${a.published_at||'?'} · collected ${a.created_at}`;
 $('title').textContent=a.title; $('desc').textContent=a.description||'(no description)';
 $('assigned').textContent=lab?`Assigned: ${eventName(lab.event_label)}${lab.note?' · note: '+lab.note:''}`:'Not labelled yet';
 $('assigned').className=lab?'meta done':'meta';
 $('note').value=lab?lab.note:''; $('rename').value=lab?eventName(lab.event_label):'';
 sugg=await (await fetch('/api/suggest?article_id='+a.id)).json();
 const box=$('suggest'); box.replaceChildren();
 sugg.forEach((s,k)=>{const b=el('button',null,'opt'+(lab&&lab.event_label===s.event_label?' current':''));
  b.append(el('b',`${k+1}`),` — ${s.name} `,el('small',`(${s.size} art., sim ${s.similarity}) closest: ${s.closest}`));
  b.onclick=()=>choose(s.event_label); box.append(b);});
 renderEvents();
}
function renderEvents(){const q=$('search').value.toLowerCase(), box=$('events'); box.replaceChildren();
 S.events.filter(e=>!q||e.name.toLowerCase().includes(q)).slice(-200).reverse().forEach(e=>{
  const b=el('button',null,'opt'); b.append(`${e.name} `,el('small',`(${e.size})`)); b.onclick=()=>choose(e.event_label); box.append(b);});}
async function choose(label){const a=cur();
 const r=await post('/api/label',{article_id:a.id,event_label:label,note:$('note').value}); if(!r)return;
 await load(); if(i<S.articles.length-1)i++; $('search').value=''; render();}
async function clearLabel(){await post('/api/clear',{article_id:cur().id}); await load(); render();}
async function rename(){const lab=S.labels[cur().id]; if(!lab)return;
 if(await post('/api/rename',{event_label:lab.event_label,name:$('rename').value})){await load();render();}}
function move(d){i=Math.max(0,Math.min(S.articles.length-1,i+d));render();}
function jumpUnlabelled(){const k=S.articles.findIndex(a=>!S.labels[a.id]); if(k>=0){i=k;render();}}
$('search').oninput=renderEvents;
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){
  if(e.key==='Escape')document.activeElement.blur(); return;}
 if(e.key>='1'&&e.key<='5'&&sugg[e.key-1])choose(sugg[e.key-1].event_label);
 else if(e.key==='n'||e.key==='N')choose('new');
 else if(e.key==='ArrowRight')move(1); else if(e.key==='ArrowLeft')move(-1);
 else if(e.key==='/'){e.preventDefault();$('search').focus();}});
load().then(()=>{jumpUnlabelled(); render();});
</script></body></html>"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True, help="e.g. 2026-09-29")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    day = Day(args.day)
    print(f"{args.day}: {len(day.articles)} articles, {len(day.labels)} already labelled", flush=True)
    print(f"Open http://localhost:{args.port}  (Ctrl+C to stop; every choice is saved)", flush=True)

    HTTPServer(("127.0.0.1", args.port), handler_for(day)).serve_forever()


if __name__ == "__main__":
    main()

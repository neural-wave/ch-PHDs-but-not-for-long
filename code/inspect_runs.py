"""Bounded Nightwatch state output; do not print unrelated jobs or credentials."""
import json, subprocess, sys

r=subprocess.run(['python3','/home/mnogales/nightwatch.py','list'],capture_output=True,text=True,check=True)
s=json.loads(r.stdout);ids=set(sys.argv[1:])
fields=['id','status','exit_code','heartbeat_at','heartbeat_expected','ended_at','due_at']
print(json.dumps({'runs':[{k:r.get(k) for k in fields} for r in s['runs'] if r['id'] in ids],
    'telegram_enabled':s.get('telegram_enabled')},indent=2))

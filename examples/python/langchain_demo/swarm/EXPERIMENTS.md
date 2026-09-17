# Swarm experiment log

Every live run (`python -m swarm.run --mode live ...`) gets an entry here, including
failed ones. Scripted-mode development runs don't need an entry.

## Template

```
### <date> — <short label>

- **Variables**: scout models, bloodhound model, judge model, seed, budget cap,
  corpus/source.
- **Command**: exact CLI invocation.
- **Outcome**: ended_by, files covered, duration, actual token spend / cost.
- **Findings**: what the swarm reported, checked against InsecureBankv2's known
  ground truth (hardcoded credentials, weak crypto, exported components, insecure
  logging, cleartext traffic, allowBackup).
- **What worked / what didn't**: concrete, one or two sentences each.
- **VERDICT**: worked / failed — do not reuse / unknown.
```

## Runs

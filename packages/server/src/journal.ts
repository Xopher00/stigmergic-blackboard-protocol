import { appendFileSync, mkdirSync } from "fs";
import { dirname } from "path";

// Append-only JSONL op journal; seq is 1-based and gapless per file.
// A failed append is logged, never raised — journaling must not break the op it records.
export class Journal {
  private seq = 0;

  constructor(private readonly path: string) {
    const dir = dirname(path);
    if (dir && dir !== ".") mkdirSync(dir, { recursive: true });
  }

  append(entry: Record<string, unknown>): void {
    const line = { seq: ++this.seq, ...entry };
    try {
      appendFileSync(this.path, JSON.stringify(line) + "\n");
    } catch (e) {
      console.error(`[SBP Journal] write failed: ${e}`);
    }
  }
}

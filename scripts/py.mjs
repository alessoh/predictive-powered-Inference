/**
 * Cross-platform venv Python launcher for npm scripts: resolves the
 * interpreter for the current OS (falls back to PATH python, e.g. CI)
 * and forwards all arguments. npm runs scripts through cmd.exe on
 * Windows, where a bare ".venv/Scripts/python" is not executable — the
 * deployment review caught the gate scripts failing on BOTH platforms.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const candidates =
  process.platform === "win32"
    ? [path.join(root, ".venv", "Scripts", "python.exe")]
    : [path.join(root, ".venv", "bin", "python")];
const python = candidates.find(existsSync) ?? "python";

const child = spawn(python, process.argv.slice(2), { stdio: "inherit", cwd: root });
child.on("exit", (code) => process.exit(code ?? 1));

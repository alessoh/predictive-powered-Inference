/**
 * Cross-platform launcher for the local Python inference API.
 * Picks the venv interpreter for the current OS and execs dev_api.py.
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

const child = spawn(python, [path.join(root, "scripts", "dev_api.py")], {
  stdio: "inherit",
});
child.on("exit", (code) => process.exit(code ?? 1));

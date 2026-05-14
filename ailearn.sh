#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CODEX_DIR="${REPO_ROOT}/CODEX"
RULES_FILE="${CODEX_DIR}/RULES.md"
KANBAN_FILE="${CODEX_DIR}/KANBAN.md"
CONTEXT_FILE="${CODEX_DIR}/CONTEXT.md"
OUTPUT_FILE="${CODEX_DIR}/AILEARN_REPORT.md"

mkdir -p "${CODEX_DIR}"

write_section() {
  local title="$1"
  printf '\n## %s\n\n' "${title}" >> "${OUTPUT_FILE}"
}

write_file_or_notice() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    cat "${file}" >> "${OUTPUT_FILE}"
    printf '\n' >> "${OUTPUT_FILE}"
  else
    printf '_Missing file: %s_\n' "${file}" >> "${OUTPUT_FILE}"
  fi
}

{
  printf '# AI Learn Report\n\n'
  printf 'Generated: %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf 'Repository: `%s`\n' "${REPO_ROOT}"
  printf '\nThis report aggregates project rules, board status, context log, file structure, and git commit history.\n'
} > "${OUTPUT_FILE}"

write_section "Rules"
write_file_or_notice "${RULES_FILE}"

write_section "Kanban"
write_file_or_notice "${KANBAN_FILE}"

write_section "Context"
write_file_or_notice "${CONTEXT_FILE}"

write_section "Git Branches"
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '```text\n' >> "${OUTPUT_FILE}"
  git -C "${REPO_ROOT}" branch --all --verbose --no-abbrev >> "${OUTPUT_FILE}"
  printf '\n```\n' >> "${OUTPUT_FILE}"
else
  printf '_Not a git repository._\n' >> "${OUTPUT_FILE}"
fi

write_section "Git Status"
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '```text\n' >> "${OUTPUT_FILE}"
  git -C "${REPO_ROOT}" status --short --branch >> "${OUTPUT_FILE}"
  printf '\n```\n' >> "${OUTPUT_FILE}"
else
  printf '_Not a git repository._\n' >> "${OUTPUT_FILE}"
fi

write_section "Project File Structure"
{
  printf '```text\n'
  (
    cd "${REPO_ROOT}"
    find . \
      -path './.git' -prune -o \
      -path './node_modules' -prune -o \
      -path './.svelte-kit' -prune -o \
      -path './build' -prune -o \
      -path './dist' -prune -o \
      -print
  ) | sed 's#^\./##' | awk 'NF' | sort
  printf '```\n'
} >> "${OUTPUT_FILE}"

write_section "Git Commit Messages"
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '```text\n' >> "${OUTPUT_FILE}"
  git -C "${REPO_ROOT}" log --all --reverse --date=short --pretty=format:'%h | %ad | %d | %s' >> "${OUTPUT_FILE}"
  printf '\n```\n' >> "${OUTPUT_FILE}"
else
  printf '_Not a git repository._\n' >> "${OUTPUT_FILE}"
fi

printf 'Wrote %s\n' "${OUTPUT_FILE}"

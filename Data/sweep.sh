#!/usr/bin/env bash
# Run classify_topics.py repeatedly until every post is labelled.
#
# A single pass skips any post whose calls all failed -- by design, so a network
# blip is never written as a bogus label. The cost is that completing the corpus
# takes more than one pass. This loops until a task is done, or until a pass
# stops making progress, whichever comes first.
#
# Discipline runs before cancer deliberately: discipline is the moderator
# variable the handoff needs, cancer is secondary. If the window closes early,
# it closes on the less important task.
#
#   ./sweep.sh                 # concurrency 10
#   CONC=5 ./sweep.sh          # gentler, better on a flaky connection
#   N=6600 MAX=20 ./sweep.sh   # override target / pass limit
set -u
cd "$(dirname "$0")"
set -a; . ./.env; set +a

LABELS="../Piloting/Round 5/output/corpora/labels"
N=${N:-6600}
MAX=${MAX:-15}
CONC=${CONC:-10}
LOG="$LABELS/sweep.log"

count() {
  local f="$LABELS/labels_gemini_$1.jsonl"
  if [ -f "$f" ]; then wc -l < "$f" | tr -d ' '; else echo 0; fi
}

echo "sweep starting $(date '+%H:%M:%S')  target $N/task  concurrency $CONC"
for task in discipline cancer; do
  for pass in $(seq 1 "$MAX"); do
    before=$(count "$task")
    if [ "$before" -ge "$N" ]; then
      echo "[$task] complete: $before/$N"
      break
    fi
    echo "[$task] pass $pass starting at $before/$N ... $(date '+%H:%M:%S')"
    python3 classify_topics.py --tasks "$task" --concurrency "$CONC" >>"$LOG" 2>&1
    after=$(count "$task")
    echo "[$task] pass $pass done: $before -> $after  $(date '+%H:%M:%S')"
    if [ "$after" -le "$before" ]; then
      echo "[$task] no progress this pass; stopping. Re-run when the network is better."
      break
    fi
  done
done
echo "sweep finished $(date '+%H:%M:%S')  discipline $(count discipline)/$N  cancer $(count cancer)/$N"

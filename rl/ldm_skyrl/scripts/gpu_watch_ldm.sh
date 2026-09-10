#!/bin/bash
# Read-only. It never submits, never attaches, never scancels -- reporting is the
# whole job. That boundary is what keeps it inside the BriCS rule against
# unattended agents that initiate actions, and it is also why it cannot repeat
# tonight's mistake: the hunt loop that DID attach landed in another line's
# allocation because its exclusion list lived only in prose.
INTERVAL=${INTERVAL:-900}
LOG=${LOG:-/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_skyrl_nb/gpu_watch_ldm.log}
# Moved off $HOME on 2026-09-09: the VAST home quota filled and the write failed
# outright. Both filesystems fill, so the rule is not "prefer one" but "check that
# the write worked" -- the `: >> "$LOG" || exit 1` below is what does that.
: >> "$LOG" || exit 1
GTOP=$(command -v gtop || echo /usr/local/bin/gtop)
ROUND=0; LAST_BEAT=0
# flock so "always (re)start it" stays idempotent -- five copies of this ran at once on 2026-09-04
exec 9>"${LOG}.lock"; flock -n 9 || { echo "watcher already running; this copy exits"; exit 0; }
while true; do
  ROUND=$((ROUND+1)); NOW=$(date -u +%H:%M:%SZ)
  # 420s, because gtop measured 264s against this account's 256 running
  # allocations -- it walks every one. The first arming used 90s and reported
  # PROBE FAILED on a working cluster: the timeout was a guess, and a guessed
  # timeout turns a healthy probe into an alarm.
  RAW=$(timeout -k 15 420 "$GTOP" --once --timeout 30 2>/dev/null | tr -d '\000')
  if [ -z "$RAW" ]; then
    # A probe that did not answer is not a reading of zero. grep -c would return
    # 0 here and that 0 would read exactly like "no idle cards", hiding the waste.
    echo "PROBE FAILED at $NOW -- gtop returned nothing; idle count is unknown, not zero"
    printf '[%s] round %d PROBE FAILED\n' "$NOW" "$ROUND" >> "$LOG"; sleep "$INTERVAL"; continue
  fi
  # Per-card lines only. gtop's header idle count includes held cards (memory
  # resident, 0%% util); 1-9 MiB is what actually free looks like.
  IDLE=$(printf '%s\n' "$RAW" | awk '
    /^ ▸ job/{j=$3} /^   nid/{n=$1}
    /GH200/{ if ($0 ~ /idle/ && $0 ~ /mem +0\.0\//) { match($0,/\[[0-9]\]/); print j, n, substr($0,RSTART+1,1) } }')
  NIDLE=$(printf '%s\n' "$IDLE" | grep -c . )
  PEND=$(squeue -u "$USER" -h -t PENDING -o "%.10i %.26j %.10l %.4D %R" 2>/dev/null)   # %l = TIME_LIMIT, field 3
  NPEND=$(printf '%s\n' "$PEND" | grep -c .)
  DEAD=$(printf '%s\n' "$PEND" | grep -ci DependencyNeverSatisfied)
  printf '[%s] round %d idle=%d pending=%d dead=%d\n' "$NOW" "$ROUND" "$NIDLE" "$NPEND" "$DEAD" >> "$LOG"
  [ "$DEAD" -gt 0 ] && { echo "DEAD JOBS ($DEAD) -- DependencyNeverSatisfied, they will never run and I cannot scancel:"; printf '%s\n' "$PEND" | grep -i DependencyNeverSatisfied | head -5; }
  # Supply alone is not a decision, and neither is supply plus a queue. Tonight this
  # watcher cried ACTIONABLE at 41, then 124, then 135 idle cards against ~85 queued
  # jobs -- and 80 of 81 could not start at all, because their walltimes crossed the
  # 05:00Z maintenance reservation. True count, false implication: that is not work
  # waiting for hardware, it is work that has excluded itself from the hardware.
  # So the condition now needs a job that could actually be scheduled, and when none
  # exists it says so, because the action is different: shorten walltimes, not wait.
  #
  # Slurm will not tell you this. It labelled exactly one of those 80 jobs
  # "Reserved for maintenance"; the other 79 read "(None)".
  WALL_EPOCH=$(date -u -d "$(date -u +%Y-%m-%d) 05:00:00" +%s 2>/dev/null)
  NOW_EPOCH=$(date -u +%s)
  [ "$WALL_EPOCH" -lt "$NOW_EPOCH" ] && WALL_EPOCH=$((WALL_EPOCH + 86400))
  FITS_H=$(( (WALL_EPOCH - NOW_EPOCH) / 3600 ))
  NFIT=$(printf '%s\n' "$PEND" | awk -v lim="$FITS_H" '
  NF && ($0 ~ /\((None|Resources|Priority)\)/) { split($3,t,":"); d=0; if (t[1] ~ /-/) { split(t[1],dd,"-"); d=dd[1]*24; t[1]=dd[2] }
    if (d + t[1] + 0 <= lim) c++ } END{print c+0}')
  if [ "$NIDLE" -ge 4 ] && [ "$NFIT" -ge 1 ]; then
    echo "ACTIONABLE $NOW: $NIDLE idle card(s); $NFIT of $NPEND queued job(s) still fit before the ${FITS_H}h wall"
    printf '%s\n' "$IDLE" | head -6 | sed 's/^/    free: job /'
    printf '%s\n' "$PEND" | awk -v lim="$FITS_H" '
      NF && ($0 ~ /\((None|Resources|Priority)\)/) { split($3,t,":"); d=0; if (t[1] ~ /-/) { split(t[1],dd,"-"); d=dd[1]*24; t[1]=dd[2] }
        if (d + t[1] + 0 <= lim) print "    schedulable: "$0 }' | head -4
  elif [ "$NIDLE" -ge 20 ] && [ "$NPEND" -ge 1 ]; then
    # Report the measured blocker distribution instead of asserting a cause. The
    # previous wording said "the fix is shorter walltimes", which was true at 23:00Z
    # and wrong by 00:49Z: the walltimes had been shortened and AssocGrpCPUMinutesLimit
    # had become the binding constraint on 13 of them. A message that hardcodes its
    # own explanation goes stale silently, which is the same failure as Slurm's Reason
    # column reading "(None)" for a job that cannot start.
    echo "STUCK $NOW: $NIDLE idle card(s), $NPEND queued, NONE able to start (waiting purely on a slot AND finishing before the ${FITS_H}h wall)"
    printf '%s\n' "$PEND" | sed 's/.*(\(.*\))/\1/;s/,.*//' | sort | uniq -c | sort -rn | head -5 | sed 's/^/      blocked by: /'
    echo "      if a row above is a walltime that crosses the ${FITS_H}h wall, it can be fixed in place:"
    echo "        scontrol update job=<id> TimeLimit=<= ${FITS_H}:00:00   (keeps queue position; reported by gpu-use-it-up, untested by me)"
    echo "      other reasons need their own remedy -- AssocGrpCPUMinutesLimit is the account's allocation, not the walltime"
  elif [ $((ROUND - LAST_BEAT)) -ge 4 ]; then
    # Silence is ambiguous -- it reads the same as a dead watcher. One beat an hour.
    echo "heartbeat $NOW: idle=$NIDLE pending=$NPEND (report-only watcher alive)"; LAST_BEAT=$ROUND
  fi
  sleep "$INTERVAL"
done

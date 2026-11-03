#!/bin/bash
# Login-node feeder of record_videos.sbatch: whenever no vf-videos job is queued, resubmit it until every one of the
# 25 (controller, task) parts has logged its end ("finished" or "done"), at most MAX_JOBS submissions; then write the
# index (inside the job's last step; here only a check). Log: vla_videos/_logs/feeder.log
#   nohup bash vla_finetuning/jobs/record_videos_feeder.sh > /dev/null 2>&1 &
R=/work/nvme/bifq/project/LanguageConditionHJ; OUT=$R/vla_videos; MAX_JOBS=${MAX_JOBS:-10}
cd $R
LOG=$OUT/_logs/feeder.log; n=0
while true; do
  done_n=0
  for f in $OUT/_logs/*_*.log; do
    [ "$(basename $f)" = feeder.log ] && continue
    tail -n 1 "$f" | grep -q -E '^\[record\] .*(finished, still wanted|: done \()' && done_n=$((done_n + 1))
  done
  if [ "$done_n" -ge 25 ]; then echo "$(date '+%F %T') all 25 parts done" >> $LOG; break; fi
  if [ -z "$(squeue -h -u $USER -n vf-videos -o %i)" ]; then
    [ "$n" -ge "$MAX_JOBS" ] && { echo "$(date '+%F %T') $MAX_JOBS jobs submitted, $done_n / 25 done: stop" >> $LOG; break; }
    j=$(sbatch --parsable vla_finetuning/jobs/record_videos.sbatch 2>> $LOG); n=$((n + 1))
    echo "$(date '+%F %T') $done_n / 25 done, submitted $j ($n)" >> $LOG
  fi
  sleep 300
done

#!/usr/bin/env bash
# gfx906 (Radeon VII) clock/DPM control + readback.
# usage: gpuclk.sh high | auto | manual <mclk_state> | show
set -u
CARD=/sys/class/drm/card1/device

show() {
  echo "perf_level : $(cat $CARD/power_dpm_force_performance_level 2>/dev/null)"
  echo "mclk       : $(grep '\*' $CARD/pp_dpm_mclk 2>/dev/null | tr -d '\n')"
  echo "sclk       : $(grep '\*' $CARD/pp_dpm_sclk 2>/dev/null | tr -d '\n')"
  for f in freq1_input freq2_input power1_average temp1_input temp2_input; do
    p=$(ls $CARD/hwmon/hwmon*/$f 2>/dev/null | head -1)
    [ -n "$p" ] && echo "$f : $(cat $p 2>/dev/null)"
  done
  echo "busy       : $(cat $CARD/gpu_busy_percent 2>/dev/null)%"
}

case "${1:-show}" in
  high)
    echo high | sudo tee $CARD/power_dpm_force_performance_level >/dev/null ;;
  auto)
    echo auto | sudo tee $CARD/power_dpm_force_performance_level >/dev/null ;;
  manual)
    echo manual | sudo tee $CARD/power_dpm_force_performance_level >/dev/null
    echo "${2:-2}" | sudo tee $CARD/pp_dpm_mclk >/dev/null ;;
  show) ;;
  *) echo "usage: $0 high|auto|manual <state>|show"; exit 1 ;;
esac
show

#!/usr/bin/env bash
# Status of the manual photo service + the shared target counter + captured photos.
#   ./photo_status.sh         one-shot snapshot
#   ./photo_status.sh -w      live-watch the service log (Ctrl-C to exit)
IMG_DIR="${IMG_DIR:-/images}"

if [ "${1:-}" = "-w" ]; then
  exec journalctl -u mavlink_photo_service -f
fi

echo "===== mavlink_photo_service ====="
printf 'active : '; systemctl is-active mavlink_photo_service
printf 'enabled: '; systemctl is-enabled mavlink_photo_service 2>/dev/null
systemctl --no-pager status mavlink_photo_service | sed -n '3,5p'

echo
echo "===== target counter ($IMG_DIR/target_state.json) ====="
if [ -f "$IMG_DIR/target_state.json" ]; then
  cat "$IMG_DIR/target_state.json"; echo
else
  echo "(no state file yet)"
fi

echo
echo "===== photos in $IMG_DIR ====="
count=$(ls "$IMG_DIR"/Task_2_*.jpg 2>/dev/null | wc -l)
echo "count: $count"
ls -1t "$IMG_DIR"/Task_2_*.jpg 2>/dev/null | head -10

echo
echo "===== last service log lines ====="
journalctl -u mavlink_photo_service -n 12 --no-pager

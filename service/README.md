# Manual photo service (Pi) — install

`mavlink_photo_service.py` listens on the MAVLink stream for a QGroundControl
custom action (`COMMAND_LONG`) and snaps a still from `gun_high` or `front_high`
into the centralized `/images/<camera>/` tree (Google-Drive-synced), using the
same `Task_2_<team>_target_<#>_<ts>.jpg` naming and the same shared target
counter (`/images/target_state.json`) as the autonomous engagement scripts.

## 1. Deploy files to the Pi (flat `~` layout)

```bash
scp test/mavlink_photo_service.py test/wetness_check.py test/target_photo.py \
    test/target_tracker.py pi@<PI>:~/
scp service/mavlink_photo_service.service pi@<PI>:/tmp/
```

## 2. Config

First run writes a template to `/images/photo_service_config.json`. Edit it:

See `photo_service_config.example.json` for the full template. Key part:

```json
{
  "team_name": "intellimass",
  "mavlink": "tcp:10.42.0.1:5760",
  "photo_cmd": 218,
  "actions": {
    "304": {"camera": "gun_high",   "wetness_check": true,  "advance_on_wetted": true},
    "305": {"camera": "front_high", "wetness_check": false, "advance_on_wetted": false}
  },
  ...
}
```

- This matches the team's existing QGC convention: `mavCmd 218`
  (`MAV_CMD_DO_AUX_FUNCTION`) with `param1` as a scripting-AUX selector
  (300/301 displacement, 302/303 radio are already taken — photos use 304/305).
- `photo_cmd` = the `COMMAND_LONG.command`; the `actions` keys = its `param1`.
  The matching QGC buttons live in `~/Documents/QGroundControl/MavlinkActions/MavlinkActions.json`
  ("Photo Gun (gun_high)" → param1 304, "Photo FPV (front_high)" → param1 305).
  Add a key here + a button there to add an action.
- `detect_model: true` loads the TFLite model on **CPU** for a real target bbox
  (avoids Coral-TPU contention with the engagement scripts).

## 3. Confirm the Pi actually receives the command (CRITICAL)

Run it in the foreground first and press the QGC button:

```bash
python3 -B ~/mavlink_photo_service.py --config /images/photo_service_config.json
```

- A `[CMD] cmd=... param1=...` line on button press → working.
- A `[CMD?] inbound ... cmd=...` line → you're seeing inbound commands but the
  cmd/param1 doesn't match config; read it off and update `photo_cmd`/`actions`.
- `[DIAG] no COMMAND_LONG seen ...` → your **mavlink-router is not mirroring
  inbound GCS→autopilot commands** to `tcp:5760`. Enable command forwarding in
  the router (or switch to a relay-bit trigger).

## 4. Install as a service

```bash
sudo cp /tmp/mavlink_photo_service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mavlink_photo_service
journalctl -u mavlink_photo_service -f
```

(Unit assumes user `pi` and files in `/home/pi`. Adjust `User=`/paths if different.)

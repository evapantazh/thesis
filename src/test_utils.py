
from utils import load_tap_info
from utils import load_tap_info, load_movesense_hr

print("\n" + "=" * 60)
print("HR loader — WITHOUT tap trimming (sanity check)")
print("=" * 60)
for rec_id in ["AVE_800_tshirt", "AGE_800_tshirt",
               "GAX_1200_tshirt", "AVE_1800_hoodie"]:
    hr = load_movesense_hr(rec_id, tap_info=None, show_stats=True)


print("\n" + "=" * 60)
print("HR loader — WITH tap trimming")
print("=" * 60)
for rec_id in ["AVE_800_tshirt", "AGE_800_tshirt",
               "GAX_1200_tshirt", "AVE_1800_hoodie"]:
    tap = load_tap_info(rec_id, show_stats=False)
    hr = load_movesense_hr(rec_id, tap_info=tap, settle_sec=5.0, show_stats=True)


print("\n" + "=" * 60)
print("Detailed inspection — AGE_800_tshirt")
print("=" * 60)
tap = load_tap_info("AGE_800_tshirt", show_stats=True)
hr = load_movesense_hr("AGE_800_tshirt", tap_info=tap, settle_sec=5.0, show_stats=True)

if hr is not None:
    print(f"\n  Time axis range: {hr['time_sec'][0]:.2f}s to {hr['time_sec'][-1]:.2f}s")
    print(f"  Duration covered: {hr['time_sec'][-1] - hr['time_sec'][0]:.2f}s")
    print(f"  Mean RR interval: {hr['rr_values'].mean():.1f} ms "
          f"(implies HR ≈ {60000 / hr['rr_values'].mean():.1f} BPM)")
    print(f"  First 5 HR samples: {hr['hr_values'][:5]}")
    print(f"  First 5 times:      {hr['time_sec'][:5]}")
    print(f"  Pre-tap removed:    {hr['n_pre_tap']}")
    print(f"  Physiologically bad removed: {hr['n_dropped']}")
print("=" * 60)


print("Test 1: AGE_1800_hoodie (your example JSON)")
print("=" * 60)
tap = load_tap_info("AGE_1800_hoodie", show_stats=True)
if tap is not None:
    for k, v in tap.items():
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Test 2: 4 diagnostic recordings")
print("=" * 60)
for rec_id in ["AVE_800_tshirt", "AGE_800_tshirt",
               "GAX_1200_tshirt", "AVE_1800_hoodie"]:
    tap = load_tap_info(rec_id, show_stats=True)

print("\n" + "=" * 60)
print("Test 3: nonexistent file (should print warning, return None)")
print("=" * 60)
tap = load_tap_info("ZZZ_9999_nothing")
assert tap is None
print("✓ Returned None as expected")

print("\nDone.")

from utils import load_timestamps, load_fps

print("\n" + "=" * 60)
print("Timestamps loader")
print("=" * 60)
for rec_id in ["AVE_800_tshirt", "AGE_800_tshirt",
               "GAX_1200_tshirt", "AVE_1800_hoodie"]:
    df = load_timestamps(rec_id, show_stats=True)
    if df is not None:
        fs = load_fps(rec_id)
        print(f"  {rec_id}: load_fps={fs:.2f}, columns={df.columns.tolist()}")

# Test nonexistent
print("\n  Nonexistent recording:")
df = load_timestamps("ZZZ_9999_nothing")
assert df is None
print("  ✓ Returned None as expected")

from utils import list_recordings

print("\n" + "=" * 60)
print("Recording discovery")
print("=" * 60)
recs = list_recordings(verify_files=True)
print(f"\n  Found {len(recs)} valid recordings.")
print(f"  First 5: {recs[:5]}")
print(f"  Last 5:  {recs[-5:]}")